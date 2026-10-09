################################################################################
#
# sloughgpt - SloughGPT shell and AI networking processor
#
################################################################################

SLOUGHPGT_VERSION = 0.1.0
SLOUGHPGT_SITE = $(SLOUGHPGT_GIT)
SLOUGHPGT_SITE_METHOD = git
SLOUGHPGT_GIT_SUBMODULES = YES
SLOUGHPGT_LICENSE = MIT
SLOUGHPGT_LICENSE_FILES = LICENSE

SLOUGHPGT_DEPENDENCIES = python3

SLOUGHPGT_MAKE_OPTS = \
	CROSS_COMPILE="$(TARGET_CROSS)" \
	CC="$(TARGET_CC)" \
	ARCH="$(TARGET_ARCH)"

define SLOUGHPGT_CONFIGURE_CMDS
	# No configure needed for pure Python package
endef

define SLOUGHPGT_BUILD_CMDS
	# Build Python package
	cd $(@D) && \
	$(TARGET_MAKE_ENV) $(PYTHON3_HOST_BIN) setup.py build \
		--build-base=$(TARGET_DIR)/usr/lib/python$(PYTHON3_VERSION_MAJOR)/site-packages
endef

define SLOUGHPGT_INSTALL_TARGET_CMDS
	# Install Python package
	cd $(@D) && \
	$(TARGET_MAKE_ENV) $(PYTHON3_HOST_BIN) setup.py install \
		--prefix=$(TARGET_DIR)/usr \
		--root=$(TARGET_DIR)

	# Install Dait shell scripts
	$(INSTALL) -D -m 0755 $(@D)/scripts/dait-shell \
		$(TARGET_DIR)/usr/bin/dait-shell

	# Install configuration
	$(INSTALL) -D -m 0644 $(@D)/config/dait.conf \
		$(TARGET_DIR)/etc/sloughgpt/dait.conf
endef

define SLOUGHPGT_INSTALL_INIT_SYSTEMD
	$(INSTALL) -D -m 0644 $(@D)/systemd/sloughgpt.service \
		$(TARGET_DIR)/etc/systemd/system/sloughgpt.service
endef

define SLOUGHPGT_INSTALL_INIT_SYSV
	$(INSTALL) -D -m 0755 $(@D)/init.d/S99sloughgpt \
		$(TARGET_DIR)/etc/init.d/S99sloughgpt
endef

$(eval $(generic-package))
