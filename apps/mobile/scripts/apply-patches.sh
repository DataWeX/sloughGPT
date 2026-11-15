#!/usr/bin/env bash
# Postinstall script – patching react-native-svg@15.11.1 for RN 0.86 Fabric API.
# Patches survive `npm install` / `yarn` by running via "postinstall" hook.
set -euo pipefail

# In-place flag: BSD sed (macOS) needs `-i ''` as two args; GNU sed (Linux)
# would treat the empty string as the script and fail with
# "sed: can't read s/...". Pick per platform — this postinstall runs on both
# dev machines and the Linux CI runner.
if sed --version >/dev/null 2>&1; then
  SED_I=(-i) # GNU
else
  SED_I=(-i '') # BSD
fi

SVG_DIR="node_modules/react-native-svg"

if [ ! -d "$SVG_DIR" ]; then
  echo "[patches] react-native-svg not found, skipping"
  exit 0
fi

echo "[patches] Applying react-native-svg × RN 0.86 compatibility patches"

# 1. RNSVGConcreteShadowNode.h — remove trailing `false` template arg (6→5 params)
#    Original: , false> {  and  , false>;
#    Patched:  > {  and  >;
sed "${SED_I[@]}" 's/, false> {/> {/g' "$SVG_DIR/common/cpp/react/renderer/components/rnsvg/RNSVGConcreteShadowNode.h"
sed "${SED_I[@]}" 's/, false>;/;>/g' "$SVG_DIR/common/cpp/react/renderer/components/rnsvg/RNSVGConcreteShadowNode.h"

# 2. RNSVGLayoutableShadowNode.cpp — StyleLength → StyleSizeLength, remove version guard
#    Original: yoga::StyleLength::points(0)  /  yoga::value::points(0)  (both guarded)
#    Patched:  yoga::StyleSizeLength::points(0)  (always, no guard)
sed "${SED_I[@]}" 's/yoga::StyleLength::points(0)/yoga::StyleSizeLength::points(0)/g' \
  "$SVG_DIR/common/cpp/react/renderer/components/rnsvg/RNSVGLayoutableShadowNode.cpp"
sed "${SED_I[@]}" 's/yoga::value::points(0)/yoga::StyleSizeLength::points(0)/g' \
  "$SVG_DIR/common/cpp/react/renderer/components/rnsvg/RNSVGLayoutableShadowNode.cpp"
# Remove the #if / #else / #endif guard
sed "${SED_I[@]}" '/#if REACT_NATIVE_MINOR_VERSION/d' \
  "$SVG_DIR/common/cpp/react/renderer/components/rnsvg/RNSVGLayoutableShadowNode.cpp"
sed "${SED_I[@]}" '/#else/d' \
  "$SVG_DIR/common/cpp/react/renderer/components/rnsvg/RNSVGLayoutableShadowNode.cpp"
sed "${SED_I[@]}" '/#endif/d' \
  "$SVG_DIR/common/cpp/react/renderer/components/rnsvg/RNSVGLayoutableShadowNode.cpp"

# 3. RNSVGImageShadowNode.h — SharedImageManager → std::shared_ptr<ImageManager>
sed "${SED_I[@]}" 's/SharedImageManager/std::shared_ptr<ImageManager>/g' \
  "$SVG_DIR/common/cpp/react/renderer/components/rnsvg/RNSVGImageShadowNode.h"

# 4. RNSVGImageComponentDescriptor.h — same
sed "${SED_I[@]}" 's/SharedImageManager/std::shared_ptr<ImageManager>/g' \
  "$SVG_DIR/common/cpp/react/renderer/components/rnsvg/RNSVGImageComponentDescriptor.h"

# 5. RNSVGImage.mm — value-type → shared_ptr for image observer
#    Instance var:  RCTImageResponseObserverProxy _imageResponseObserverProxy;
#    →              std::shared_ptr<RCTImageResponseObserverProxy> _imageResponseObserverProxy;
#    Init:          = RCTImageResponseObserverProxy(self)
#    →              = std::make_shared<RCTImageResponseObserverProxy>(self)
sed "${SED_I[@]}" 's/RCTImageResponseObserverProxy _imageResponseObserverProxy;/std::shared_ptr<RCTImageResponseObserverProxy> _imageResponseObserverProxy;/' \
  "$SVG_DIR/apple/Elements/RNSVGImage.mm"
sed "${SED_I[@]}" 's/= RCTImageResponseObserverProxy(self);/= std::make_shared<RCTImageResponseObserverProxy>(self);/' \
  "$SVG_DIR/apple/Elements/RNSVGImage.mm"

echo "[patches] Done"
