//! Streaming response compression for the edge.
//!
//! Stateless passthrough: negotiate Accept-Encoding, stream-compress body
//! chunks, never buffer the full body, never touch disk. Identity fallback
//! when the client can't decode, the payload is already encoded, the
//! response is SSE/206/no-transform, or the body is tiny.

use axum::body::Body;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use flate2::write::GzEncoder;
use flate2::Compression;
use futures::{Stream, StreamExt};
use std::io::Write;

/// Skip compression for these content-type prefixes.
/// Note: `application/octet-stream` is intentionally NOT skipped — it is the
/// primary file-download type this gateway exists to compress.
const SKIP_CT_PREFIXES: [&str; 6] = [
    "text/event-stream",
    "image/",
    "video/",
    "audio/",
    "application/zip",
    "application/gzip",
];

/// Known bodies smaller than this stream through identity — compression
/// overhead isn't worth it.
const MIN_COMPRESS_BYTES: u64 = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Zstd,
    Gzip,
}

impl Codec {
    pub fn as_str(self) -> &'static str {
        match self {
            Codec::Zstd => "zstd",
            Codec::Gzip => "gzip",
        }
    }
}

/// Pick a codec from the request's Accept-Encoding, or None → identity.
pub fn negotiate(client_encoding: Option<&HeaderValue>) -> Option<Codec> {
    let raw = client_encoding?.to_str().ok()?.to_ascii_lowercase();
    if accepts(&raw, "zstd") {
        return Some(Codec::Zstd);
    }
    if accepts(&raw, "gzip") {
        return Some(Codec::Gzip);
    }
    None
}

/// True when *token* appears in the header with a non-zero q-value.
fn accepts(header: &str, token: &str) -> bool {
    header.split(',').map(str::trim).any(|part| {
        let mut it = part.splitn(2, ';');
        let name = it.next().unwrap_or("").trim();
        if !name.eq_ignore_ascii_case(token) && name != "*" {
            return false;
        }
        match it.next() {
            None => true,
            Some(rest) => {
                // q=0 → rejected; missing/invalid q or q>0 → accepted.
                let rejected = rest.split(';').filter_map(|p| p.trim().split_once('=')).any(
                    |(k, v)| {
                        k.trim().eq_ignore_ascii_case("q")
                            && v.trim()
                                .parse::<f32>()
                                .map(|q| q <= 0.0)
                                .unwrap_or(false)
                    },
                );
                !rejected
            }
        }
    })
}

/// True when this response must stream through untouched (identity passthrough).
pub fn should_skip_response(
    status: StatusCode,
    req_headers: &HeaderMap,
    resp_headers: &HeaderMap,
) -> bool {
    // Already encoded upstream — never double-compress.
    if resp_headers.contains_key(header::CONTENT_ENCODING) {
        return true;
    }
    // 206 / Range: byte offsets address the identity body only.
    if status == StatusCode::PARTIAL_CONTENT || req_headers.contains_key(header::RANGE) {
        return true;
    }
    // Explicit opt-out.
    if let Some(cc) = resp_headers.get(header::CACHE_CONTROL) {
        if cc.to_str().map(|s| s.contains("no-transform")).unwrap_or(false) {
            return true;
        }
    }
    let ct = resp_headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if SKIP_CT_PREFIXES.iter().any(|p| ct.starts_with(p)) {
        return true;
    }
    // Only success bodies benefit; errors stay identity for easy debugging.
    if !status.is_success() {
        return true;
    }
    if let Some(len) = resp_headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
    {
        if len < MIN_COMPRESS_BYTES {
            return true;
        }
    }
    false
}

// ── Streaming sink ──────────────────────────────────────────────────────────

enum Sink {
    Gzip(GzEncoder<Vec<u8>>),
    Zstd(zstd::stream::write::Encoder<'static, Vec<u8>>),
}

impl Sink {
    fn new(codec: Codec, known_len: Option<u64>) -> std::io::Result<Self> {
        match codec {
            Codec::Gzip => Ok(Sink::Gzip(GzEncoder::new(
                Vec::new(),
                Compression::default(),
            ))),
            Codec::Zstd => {
                let mut enc = zstd::stream::write::Encoder::new(Vec::new(), 3)?;
                // When identity Content-Length is known, stamp it into the
                // frame header so strict one-shot decompressors work.
                if let Some(len) = known_len {
                    enc.set_pledged_src_size(Some(len))?;
                    enc.include_contentsize(true)?;
                }
                Ok(Sink::Zstd(enc))
            }
        }
    }

    fn write_chunk(&mut self, data: &[u8]) -> std::io::Result<Vec<u8>> {
        match self {
            Sink::Gzip(enc) => {
                enc.write_all(data)?;
                enc.flush()?;
                Ok(std::mem::take(enc.get_mut()))
            }
            Sink::Zstd(enc) => {
                enc.write_all(data)?;
                enc.flush()?;
                Ok(std::mem::take(enc.get_mut()))
            }
        }
    }

    fn finish(self) -> std::io::Result<Vec<u8>> {
        match self {
            Sink::Gzip(enc) => enc.finish(),
            Sink::Zstd(enc) => enc.finish(),
        }
    }
}

/// Stream-compress *upstream* with *codec*. Yields encoded chunks as they
/// are produced — the full body is never held in memory.
///
/// `known_len`: identity Content-Length when available (stamped into zstd
/// frame headers; ignored by gzip).
pub fn compress_stream<S, E>(
    upstream: S,
    codec: Codec,
    known_len: Option<u64>,
) -> impl Stream<Item = Result<Bytes, std::io::Error>>
where
    S: Stream<Item = Result<Bytes, E>>,
    E: std::fmt::Display,
{
    async_stream::try_stream! {
        let mut sink = Sink::new(codec, known_len)?;
        futures::pin_mut!(upstream);
        while let Some(item) = upstream.next().await {
            let chunk = match item {
                Ok(b) => b,
                Err(e) => Err(std::io::Error::other(e.to_string()))?,
            };
            if chunk.is_empty() {
                continue;
            }
            let out = sink.write_chunk(&chunk)?;
            if !out.is_empty() {
                yield Bytes::from(out);
            }
        }
        let tail = sink.finish()?;
        if !tail.is_empty() {
            yield Bytes::from(tail);
        }
    }
}

/// Apply negotiated compression to a relayed response.
/// Identity passthrough when negotiation/skip rules say so.
pub fn maybe_compress_response(
    resp: Response,
    codec: Option<Codec>,
    req_headers: &HeaderMap,
) -> Response {
    let (parts, body) = resp.into_parts();

    let codec = match codec {
        Some(c) if !should_skip_response(parts.status, req_headers, &parts.headers) => c,
        _ => return Response::from_parts(parts, body),
    };

    let known_len = parts
        .headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let upstream = body.into_data_stream();
    let compressed = compress_stream(upstream, codec, known_len);

    let mut headers = parts.headers;
    headers.insert(
        header::CONTENT_ENCODING,
        HeaderValue::from_str(codec.as_str()).expect("static codec name"),
    );
    // Length of the identity body is no longer valid.
    headers.remove(header::CONTENT_LENGTH);
    headers.insert(header::VARY, HeaderValue::from_static("Accept-Encoding"));

    let mut out = Response::new(Body::from_stream(compressed));
    *out.status_mut() = parts.status;
    *out.headers_mut() = headers;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn hv(s: &str) -> HeaderValue {
        HeaderValue::from_str(s).unwrap()
    }

    fn sample_body() -> Vec<u8> {
        // Highly compressible “file download” payload.
        let mut v = Vec::with_capacity(256 * 1024);
        for i in 0..4096u32 {
            v.extend_from_slice(format!("record {i:08} payload padding padding\n").as_bytes());
        }
        v
    }

    async fn collect(stream: impl Stream<Item = Result<Bytes, std::io::Error>>) -> Vec<u8> {
        let mut out = Vec::new();
        futures::pin_mut!(stream);
        while let Some(chunk) = stream.next().await {
            out.extend_from_slice(&chunk.unwrap());
        }
        out
    }

    async fn body_bytes(resp: Response) -> Vec<u8> {
        let mut stream = resp.into_body().into_data_stream();
        let mut out = Vec::new();
        while let Some(next) = stream.next().await {
            out.extend_from_slice(&next.unwrap());
        }
        out
    }

    #[test]
    fn negotiate_prefers_zstd_then_gzip() {
        assert_eq!(negotiate(Some(&hv("gzip, zstd"))), Some(Codec::Zstd));
        assert_eq!(negotiate(Some(&hv("gzip"))), Some(Codec::Gzip));
        assert_eq!(negotiate(Some(&hv("zstd;q=0, gzip"))), Some(Codec::Gzip));
        assert_eq!(negotiate(Some(&hv("gzip;q=0"))), None);
        assert_eq!(negotiate(Some(&hv("identity"))), None);
        assert_eq!(negotiate(None), None);
        assert_eq!(negotiate(Some(&hv("*"))), Some(Codec::Zstd));
    }

    #[test]
    fn skip_already_compressed_sse_range_small() {
        let req = HeaderMap::new();
        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_ENCODING, hv("gzip"));
        assert!(should_skip_response(StatusCode::OK, &req, &resp));

        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_TYPE, hv("text/event-stream"));
        assert!(should_skip_response(StatusCode::OK, &req, &resp));

        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_TYPE, hv("application/json"));
        assert!(should_skip_response(
            StatusCode::PARTIAL_CONTENT,
            &req,
            &resp
        ));

        let mut req = HeaderMap::new();
        req.insert(header::RANGE, hv("bytes=0-100"));
        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_TYPE, hv("application/json"));
        resp.insert(header::CONTENT_LENGTH, hv("100000"));
        assert!(should_skip_response(StatusCode::OK, &req, &resp));

        let req = HeaderMap::new();
        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_TYPE, hv("application/json"));
        resp.insert(header::CONTENT_LENGTH, hv("10"));
        assert!(should_skip_response(StatusCode::OK, &req, &resp));

        // Eligible large JSON
        let req = HeaderMap::new();
        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_TYPE, hv("application/json"));
        resp.insert(header::CONTENT_LENGTH, hv("100000"));
        assert!(!should_skip_response(StatusCode::OK, &req, &resp));

        // File downloads (octet-stream) MUST compress — that's the point.
        let req = HeaderMap::new();
        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_TYPE, hv("application/octet-stream"));
        resp.insert(header::CONTENT_LENGTH, hv("100000"));
        assert!(!should_skip_response(StatusCode::OK, &req, &resp));
    }

    #[tokio::test]
    async fn round_trip_gzip_and_zstd() {
        let payload = sample_body();
        for codec in [Codec::Gzip, Codec::Zstd] {
            let chunks: Vec<Result<Bytes, std::io::Error>> = payload
                .chunks(7 * 1024)
                .map(|c| Ok(Bytes::copy_from_slice(c)))
                .collect();
            let encoded =
                collect(compress_stream(futures::stream::iter(chunks), codec, Some(payload.len() as u64)))
                    .await;
            assert!(encoded.len() < payload.len(), "{codec:?} must shrink");

            let decoded = match codec {
                Codec::Gzip => {
                    use flate2::read::GzDecoder;
                    use std::io::Read;
                    let mut d = GzDecoder::new(&encoded[..]);
                    let mut out = Vec::new();
                    d.read_to_end(&mut out).unwrap();
                    out
                }
                Codec::Zstd => zstd::decode_all(&encoded[..]).unwrap(),
            };
            assert_eq!(decoded, payload, "lossless round-trip {codec:?}");
        }
    }

    #[tokio::test]
    async fn identity_passthrough_when_no_codec() {
        let payload = sample_body();
        let resp = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, payload.len())
            .body(Body::from(payload.clone()))
            .unwrap();
        let req = HeaderMap::new();
        let out = maybe_compress_response(resp, None, &req);
        assert!(!out.headers().contains_key(header::CONTENT_ENCODING));
        assert_eq!(body_bytes(out).await, payload);
    }

    #[tokio::test]
    async fn compress_sets_headers_and_shrinks() {
        let payload = sample_body();
        let resp = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, payload.len())
            .body(Body::from(payload.clone()))
            .unwrap();
        let req = HeaderMap::new();
        let out = maybe_compress_response(resp, Some(Codec::Gzip), &req);
        assert_eq!(
            out.headers().get(header::CONTENT_ENCODING).unwrap(),
            "gzip"
        );
        assert!(out.headers().get(header::CONTENT_LENGTH).is_none());
        assert_eq!(out.headers().get(header::VARY).unwrap(), "Accept-Encoding");
        let wire = body_bytes(out).await;
        assert!(wire.len() < payload.len());

        use flate2::read::GzDecoder;
        use std::io::Read;
        let mut d = GzDecoder::new(&wire[..]);
        let mut decoded = Vec::new();
        d.read_to_end(&mut decoded).unwrap();
        assert_eq!(decoded, payload);
    }

    #[tokio::test]
    async fn already_encoded_passthrough_byte_identical() {
        let payload = sample_body();
        let resp = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_ENCODING, "gzip")
            .header(header::CONTENT_LENGTH, payload.len())
            .body(Body::from(payload.clone()))
            .unwrap();
        let req = HeaderMap::new();
        let out = maybe_compress_response(resp, Some(Codec::Zstd), &req);
        assert_eq!(
            out.headers().get(header::CONTENT_ENCODING).unwrap(),
            "gzip"
        );
        assert_eq!(body_bytes(out).await, payload);
    }

    #[tokio::test]
    async fn range_request_passthrough() {
        let payload = sample_body();
        let resp = Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_RANGE, "bytes 0-99/100000")
            .body(Body::from(payload.clone()))
            .unwrap();
        let mut req = HeaderMap::new();
        req.insert(header::RANGE, hv("bytes=0-99"));
        let out = maybe_compress_response(resp, Some(Codec::Gzip), &req);
        assert!(!out.headers().contains_key(header::CONTENT_ENCODING));
        assert_eq!(body_bytes(out).await, payload);
    }

    #[tokio::test]
    async fn octet_stream_download_compresses_losslessly() {
        let payload = sample_body();
        let resp = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::CONTENT_LENGTH, payload.len())
            .body(Body::from(payload.clone()))
            .unwrap();
        let req = HeaderMap::new();
        let out = maybe_compress_response(resp, Some(Codec::Zstd), &req);
        assert_eq!(
            out.headers().get(header::CONTENT_ENCODING).unwrap(),
            "zstd"
        );
        let wire = body_bytes(out).await;
        assert!(wire.len() < payload.len() / 2);
        let decoded = zstd::decode_all(&wire[..]).unwrap();
        assert_eq!(decoded, payload);
    }
}
