require "json"

package = JSON.parse(File.read(File.join(__dir__, "../package.json")))

Pod::Spec.new do |s|
  s.name         = "SloNet"
  s.version      = package["version"]
  s.summary      = "SloNet on-device inference engine"
  s.homepage     = "https://github.com/sloughgpt/sloughgpt"
  s.license      = "MIT"
  s.authors      = "SloughGPT Team"
  s.platforms    = { :ios => "15.0" }
  s.source       = { :git => "https://github.com/sloughgpt/sloughgpt.git", :tag => "v#{s.version}" }
  s.source_files = "slonet.h", "slonet.c"
  s.frameworks   = "Accelerate"
  s.libraries    = "c++"
  s.pod_target_xcconfig = {
    "OTHER_CFLAGS" => "-O3 -ffast-math",
  }
end
