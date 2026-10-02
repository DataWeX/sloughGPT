//! Streaming response compression for the edge.
//!
//! Stateless passthrough: negotiate Accept-Encoding, stream-compress body
//! chunks, never buffer the full body, never touch disk. Identity fallback
//! when the client can't decode, the payload is already encoded, the
//! response is 206/no-transform, or a *known* body is tiny.
//!
//! SSE (`text/event-stream`) compresses too — it is the model's token
//! stream, the largest sustained bandwidth item during inference. SSE
//! carries no Content-Length, so it starts encoding at byte 0: no peek,
//! no hold, first-token latency untouched. A/B for benchmarks: send
//! `Accept-Encoding` → gzip/zstd, omit it → identity.

use axum::body::Body;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use flate2::write::GzEncoder;
use flate2::Compression;
use futures::{Stream, StreamExt};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;

// ── Bandwidth observability ─────────────────────────────────────────────────
//
// The edge owns the encode path, so it is the single source of truth for
// byte counts. Python never re-counts: it scrapes the `bandwidth` block of
// GET /health/detailed and logs interval deltas through the infra logger.
// Counting is two relaxed atomic adds per chunk on the hot path — no locks,
// no allocation, no effect on first-byte latency.

/// Process-wide byte counters for relayed traffic (relayed responses only;
/// gateway-owned routes like `/health/*` and static files never pass
/// through `maybe_compress_response`).
#[derive(Debug, Default)]
pub struct BandwidthCounters {
    /// Uncompressed payload bytes presented to clients (input to the
    /// encoder for compressed responses; identical to `wire_bytes` for
    /// identity passthrough).
    pub identity_bytes: AtomicU64,
    /// Actual bytes emitted on the wire (encoded output, or passthrough
    /// bytes unchanged).
    pub wire_bytes: AtomicU64,
    pub compressed_responses: AtomicU64,
    pub identity_responses: AtomicU64,
    pub zstd_responses: AtomicU64,
    pub gzip_responses: AtomicU64,
}

impl BandwidthCounters {
    pub const fn new() -> Self {
        Self {
            identity_bytes: AtomicU64::new(0),
            wire_bytes: AtomicU64::new(0),
            compressed_responses: AtomicU64::new(0),
            identity_responses: AtomicU64::new(0),
            zstd_responses: AtomicU64::new(0),
            gzip_responses: AtomicU64::new(0),
        }
    }

    /// Point-in-time read for /health/detailed. Field loads are independent
    /// (Relaxed) — mid-stream values are expected and fine for observability.
    pub fn snapshot(&self) -> BandwidthSnapshot {
        let identity = self.identity_bytes.load(Relaxed);
        let wire = self.wire_bytes.load(Relaxed);
        // Signed: an incompressible payload can expand on the wire, and
        // hiding that would make savings lie.
        let saved = identity as i64 - wire as i64;
        let saved_pct = if identity == 0 {
            0.0
        } else {
            saved as f64 * 100.0 / identity as f64
        };
        BandwidthSnapshot {
            identity_bytes: identity,
            wire_bytes: wire,
            saved_bytes: saved,
            saved_pct,
            compressed_responses: self.compressed_responses.load(Relaxed),
            identity_responses: self.identity_responses.load(Relaxed),
            zstd_responses: self.zstd_responses.load(Relaxed),
            gzip_responses: self.gzip_responses.load(Relaxed),
        }
    }
}

/// Serializable view served on `GET /health/detailed` under `bandwidth`.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct BandwidthSnapshot {
    pub identity_bytes: u64,
    pub wire_bytes: u64,
    pub saved_bytes: i64,
    pub saved_pct: f64,
    pub compressed_responses: u64,
    pub identity_responses: u64,
    pub zstd_responses: u64,
    pub gzip_responses: u64,
}

/// The process-wide counters (lazily initialized, `'static` so response
/// streams can borrow them without an `Arc`).
pub fn bandwidth() -> &'static BandwidthCounters {
    static COUNTERS: OnceLock<BandwidthCounters> = OnceLock::new();
    COUNTERS.get_or_init(BandwidthCounters::new)
}

/// Skip compression for these content-type prefixes.
/// Note: `application/octet-stream` is intentionally NOT skipped — it is the
/// primary file-download type this gateway exists to compress. Nor is
/// `text/event-stream`: the token stream *is* the bandwidth we're saving.
const SKIP_CT_PREFIXES: [&str; 5] = [
    "image/",
    "video/",
    "audio/",
    "application/zip",
    "application/gzip",
];

/// Known bodies smaller than this stream through identity — compression
/// overhead isn't worth it. Unknown-length bodies (SSE) never gate on it:
/// they are open-ended streams, not tiny responses.
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
                let rejected = rest
                    .split(';')
                    .filter_map(|p| p.trim().split_once('='))
                    .any(|(k, v)| {
                        k.trim().eq_ignore_ascii_case("q")
                            && v.trim().parse::<f32>().map(|q| q <= 0.0).unwrap_or(false)
                    });
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
        if cc
            .to_str()
            .map(|s| s.contains("no-transform"))
            .unwrap_or(false)
        {
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
    // 204/205 have no body — compressing would emit a useless empty frame.
    if !status.is_success()
        || status == StatusCode::NO_CONTENT
        || status == StatusCode::RESET_CONTENT
    {
        return true;
    }
    // Size gate applies to *known* lengths only: an unknown-length body
    // (SSE token stream) is open-ended and compresses from byte 0.
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
    maybe_compress_response_with(resp, codec, req_headers, bandwidth())
}

/// Counting core: identical to [`maybe_compress_response`] but records byte
/// counts into `counters`. Tests pass a function-local `static` for
/// isolation; production passes the process-wide counters.
fn maybe_compress_response_with(
    resp: Response,
    codec: Option<Codec>,
    req_headers: &HeaderMap,
    counters: &'static BandwidthCounters,
) -> Response {
    let (parts, body) = resp.into_parts();

    let codec = match codec {
        Some(c) if !should_skip_response(parts.status, req_headers, &parts.headers) => c,
        _ => {
            // Identity passthrough: count real bytes as they cross the
            // wire — identity == wire here, so savings stay honest.
            counters.identity_responses.fetch_add(1, Relaxed);
            let counted = body.into_data_stream().map(move |chunk| {
                if let Ok(b) = &chunk {
                    let n = b.len() as u64;
                    counters.identity_bytes.fetch_add(n, Relaxed);
                    counters.wire_bytes.fetch_add(n, Relaxed);
                }
                chunk
            });
            return Response::from_parts(parts, Body::from_stream(counted));
        }
    };

    counters.compressed_responses.fetch_add(1, Relaxed);
    match codec {
        Codec::Zstd => counters.zstd_responses.fetch_add(1, Relaxed),
        Codec::Gzip => counters.gzip_responses.fetch_add(1, Relaxed),
    };

    let known_len = parts
        .headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    // identity_bytes: what the client would have received uncompressed —
    // counted at the encoder input, exact even for unknown-length SSE.
    let upstream = body.into_data_stream().map(move |chunk| {
        if let Ok(b) = &chunk {
            counters.identity_bytes.fetch_add(b.len() as u64, Relaxed);
        }
        chunk
    });
    // wire_bytes: what actually leaves the process — counted at the
    // encoder output.
    let compressed = compress_stream(upstream, codec, known_len).map(move |chunk| {
        if let Ok(b) = &chunk {
            counters.wire_bytes.fetch_add(b.len() as u64, Relaxed);
        }
        chunk
    });

    let mut headers = parts.headers;
    headers.insert(
        header::CONTENT_ENCODING,
        HeaderValue::from_str(codec.as_str()).expect("static codec name"),
    );
    // Advertise identity size so clients can drive progress bars / resume
    // accounting against the decoded body, not the wire size.
    if let Some(len) = known_len {
        if let Ok(v) = HeaderValue::from_str(&len.to_string()) {
            headers.insert("x-uncompressed-content-length", v);
        }
    }
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
    fn skip_already_compressed_range_small() {
        let req = HeaderMap::new();
        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_ENCODING, hv("gzip"));
        assert!(should_skip_response(StatusCode::OK, &req, &resp));

        // SSE is NOT skipped: the token stream compresses from byte 0
        // (no Content-Length → no size gate).
        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_TYPE, hv("text/event-stream; charset=utf-8"));
        assert!(!should_skip_response(StatusCode::OK, &req, &resp));

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

        // Empty/204 never benefits from a compressed empty frame.
        let req = HeaderMap::new();
        let mut resp = HeaderMap::new();
        resp.insert(header::CONTENT_TYPE, hv("application/json"));
        assert!(should_skip_response(StatusCode::NO_CONTENT, &req, &resp));
    }

    #[tokio::test]
    async fn round_trip_gzip_and_zstd() {
        let payload = sample_body();
        for codec in [Codec::Gzip, Codec::Zstd] {
            let chunks: Vec<Result<Bytes, std::io::Error>> = payload
                .chunks(7 * 1024)
                .map(|c| Ok(Bytes::copy_from_slice(c)))
                .collect();
            let encoded = collect(compress_stream(
                futures::stream::iter(chunks),
                codec,
                Some(payload.len() as u64),
            ))
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
        assert_eq!(out.headers().get(header::CONTENT_ENCODING).unwrap(), "gzip");
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
    async fn compressed_response_advertises_identity_length() {
        let payload = sample_body();
        let resp = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::CONTENT_LENGTH, payload.len())
            .body(Body::from(payload.clone()))
            .unwrap();
        let req = HeaderMap::new();
        let out = maybe_compress_response(resp, Some(Codec::Gzip), &req);
        let hdr = out
            .headers()
            .get("x-uncompressed-content-length")
            .expect("identity length header");
        assert_eq!(hdr.to_str().unwrap(), payload.len().to_string());
        assert!(out.headers().get(header::CONTENT_LENGTH).is_none());
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
        assert_eq!(out.headers().get(header::CONTENT_ENCODING).unwrap(), "gzip");
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
        assert_eq!(out.headers().get(header::CONTENT_ENCODING).unwrap(), "zstd");
        let wire = body_bytes(out).await;
        assert!(wire.len() < payload.len() / 2);
        let decoded = zstd::decode_all(&wire[..]).unwrap();
        assert_eq!(decoded, payload);
    }

    /// Synthetic model token stream: repetitive SSE `data:` frames, no framing.
    fn sse_events(bytes: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(bytes + 128);
        let mut i = 0u32;
        while v.len() < bytes {
            v.extend_from_slice(
                format!(
                    "data: {{\"type\":\"delta\",\"idx\":{i:06},\"content\":\"token payload padding padding\"}}\n\n"
                )
                .as_bytes(),
            );
            i += 1;
        }
        v
    }

    fn sse_response(payload: Vec<u8>) -> Response {
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
            .body(Body::from(payload))
            .unwrap() // no Content-Length — like a live token stream
    }

    fn decode(codec: Codec, wire: &[u8]) -> Vec<u8> {
        match codec {
            Codec::Gzip => {
                use flate2::read::GzDecoder;
                use std::io::Read;
                let mut d = GzDecoder::new(wire);
                let mut out = Vec::new();
                d.read_to_end(&mut out).unwrap();
                out
            }
            Codec::Zstd => zstd::decode_all(wire).unwrap(),
        }
    }

    #[tokio::test]
    async fn sse_token_stream_compresses_losslessly() {
        let payload = sse_events(512 * 1024);
        for codec in [Codec::Gzip, Codec::Zstd] {
            let out = maybe_compress_response(
                sse_response(payload.clone()),
                Some(codec),
                &HeaderMap::new(),
            );
            assert_eq!(
                out.headers().get(header::CONTENT_ENCODING).unwrap(),
                codec.as_str()
            );
            assert!(out.headers().get(header::CONTENT_LENGTH).is_none());
            assert_eq!(out.headers().get(header::VARY).unwrap(), "Accept-Encoding");
            assert!(out.headers().get("x-uncompressed-content-length").is_none());
            let wire = body_bytes(out).await;
            assert!(
                wire.len() < payload.len() / 2,
                "{codec:?} must shrink the token stream ({} → {})",
                payload.len(),
                wire.len()
            );
            assert_eq!(decode(codec, &wire), payload, "lossless {codec:?}");
        }
    }

    #[tokio::test]
    async fn tiny_sse_compresses_from_byte_zero_no_size_gate() {
        let payload = sse_events(200);
        let out = maybe_compress_response(
            sse_response(payload.clone()),
            Some(Codec::Gzip),
            &HeaderMap::new(),
        );
        // Unknown length → the tiny-body gate never applies; encoding starts
        // at byte 0 so the first token frame is never held back.
        assert_eq!(out.headers().get(header::CONTENT_ENCODING).unwrap(), "gzip");
        let wire = body_bytes(out).await;
        assert_eq!(decode(Codec::Gzip, &wire), payload);
    }

    #[tokio::test]
    async fn sse_without_accept_encoding_stays_identity() {
        let payload = sse_events(4096);
        let out = maybe_compress_response(sse_response(payload.clone()), None, &HeaderMap::new());
        assert!(!out.headers().contains_key(header::CONTENT_ENCODING));
        assert_eq!(body_bytes(out).await, payload);
    }

    // ── Bandwidth counters ───────────────────────────────────────────────

    #[tokio::test]
    async fn counters_track_compressed_response() {
        static C: BandwidthCounters = BandwidthCounters::new();
        let payload = sample_body();
        let resp = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::CONTENT_LENGTH, payload.len())
            .body(Body::from(payload.clone()))
            .unwrap();
        let out = maybe_compress_response_with(resp, Some(Codec::Zstd), &HeaderMap::new(), &C);
        let wire = body_bytes(out).await;

        let snap = C.snapshot();
        assert_eq!(snap.compressed_responses, 1);
        assert_eq!(snap.zstd_responses, 1);
        assert_eq!(snap.gzip_responses, 0);
        assert_eq!(snap.identity_responses, 0);
        // Identity counted at encoder input, wire at encoder output.
        assert_eq!(snap.identity_bytes, payload.len() as u64);
        assert_eq!(snap.wire_bytes, wire.len() as u64);
        assert!(snap.saved_bytes > 0, "zstd must save bytes");
        assert!(snap.saved_pct > 50.0);
    }

    #[tokio::test]
    async fn counters_track_identity_passthrough() {
        static C: BandwidthCounters = BandwidthCounters::new();
        let payload = sample_body();
        let resp = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, payload.len())
            .body(Body::from(payload.clone()))
            .unwrap();
        let out = maybe_compress_response_with(resp, None, &HeaderMap::new(), &C);
        assert_eq!(body_bytes(out).await, payload);

        let snap = C.snapshot();
        assert_eq!(snap.identity_responses, 1);
        assert_eq!(snap.compressed_responses, 0);
        // Passthrough: wire == identity, savings zero — must not fabricate.
        assert_eq!(snap.identity_bytes, payload.len() as u64);
        assert_eq!(snap.wire_bytes, payload.len() as u64);
        assert_eq!(snap.saved_bytes, 0);
        assert_eq!(snap.saved_pct, 0.0);
    }

    #[tokio::test]
    async fn counters_dont_track_uncompressed_direct_calls() {
        // compress_stream itself stays counter-free: counting lives only in
        // maybe_compress_response so pure encode benchmarks/tests are inert.
        static C: BandwidthCounters = BandwidthCounters::new();
        let payload = sample_body();
        let chunks: Vec<Result<Bytes, std::io::Error>> = payload
            .chunks(7 * 1024)
            .map(|c| Ok(Bytes::copy_from_slice(c)))
            .collect();
        let encoded = collect(
            compress_stream(futures::stream::iter(chunks), Codec::Gzip, None),
        )
        .await;
        assert!(!encoded.is_empty());
        assert_eq!(C.snapshot().wire_bytes, 0);
        assert_eq!(C.snapshot().identity_bytes, 0);
    }

    #[test]
    fn snapshot_reports_expansion_as_negative_savings() {
        // Incompressible/tiny payload where wire > identity must not lie.
        let c = BandwidthCounters::new();
        c.identity_bytes.store(100, Relaxed);
        c.wire_bytes.store(120, Relaxed);
        let snap = c.snapshot();
        assert_eq!(snap.saved_bytes, -20);
        assert!((snap.saved_pct - (-20.0)).abs() < 1e-9);

        let empty = BandwidthCounters::new().snapshot();
        assert_eq!(empty.saved_pct, 0.0, "no traffic → 0%, not NaN");
        assert_eq!(empty.identity_bytes, 0);
    }
}
