"""Tests for Buildroot Docker configuration."""
import os
import stat

import pytest

# Buildroot is at repo root, not relative to this test file
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
_BUILDROOT = os.path.join(_REPO_ROOT, "buildroot")


@pytest.mark.slow
class TestBuildrootDocker:
    """Tests for Buildroot Docker build configuration."""

    def test_dockerfile_exists(self):
        """Dockerfile exists in buildroot directory."""
        dockerfile = os.path.join(_BUILDROOT, "Dockerfile")
        assert os.path.isfile(dockerfile), f"Dockerfile not found at {dockerfile}"

    def test_dockerfile_has_from_instruction(self):
        """Dockerfile has a FROM instruction."""
        dockerfile = os.path.join(_BUILDROOT, "Dockerfile")
        with open(dockerfile) as f:
            content = f.read()
        assert "FROM" in content, "Dockerfile missing FROM instruction"

    def test_build_script_exists(self):
        """build.sh exists and is executable."""
        build_script = os.path.join(_BUILDROOT, "build.sh")
        assert os.path.isfile(build_script), f"build.sh not found at {build_script}"
        st = os.stat(build_script)
        assert st.st_mode & stat.S_IXUSR, "build.sh is not executable"

    def test_build_script_has_required_functions(self):
        """build.sh has required build functions."""
        build_script = os.path.join(_BUILDROOT, "build.sh")
        with open(build_script) as f:
            content = f.read()
        assert "docker" in content.lower() or "buildroot" in content.lower(), \
            "build.sh should reference docker or buildroot"

    def test_build_script_references_correct_image(self):
        """build.sh references correct Docker image name."""
        build_script = os.path.join(_BUILDROOT, "build.sh")
        with open(build_script) as f:
            content = f.read()
        assert "buildroot" in content.lower(), \
            "build.sh should reference buildroot image"

    def test_dockerfile_has_workdir(self):
        """Dockerfile has a WORKDIR instruction."""
        dockerfile = os.path.join(_BUILDROOT, "Dockerfile")
        with open(dockerfile) as f:
            content = f.read()
        assert "WORKDIR" in content, "Dockerfile missing WORKDIR instruction"

    def test_dockerfile_has_copy_or_add(self):
        """Dockerfile has COPY or ADD instruction."""
        dockerfile = os.path.join(_BUILDROOT, "Dockerfile")
        with open(dockerfile) as f:
            content = f.read()
        has_copy = "COPY" in content
        has_add = "ADD" in content
        assert has_copy or has_add, "Dockerfile missing COPY or ADD instruction"

    def test_configs_directory_exists(self):
        """configs directory exists for Buildroot."""
        configs_dir = os.path.join(_BUILDROOT, "configs")
        assert os.path.isdir(configs_dir), f"configs directory not found at {configs_dir}"

    def test_overlays_directory_exists(self):
        """overlays directory exists for Buildroot."""
        overlays_dir = os.path.join(_BUILDROOT, "overlays")
        assert os.path.isdir(overlays_dir), f"overlays directory not found at {overlays_dir}"
