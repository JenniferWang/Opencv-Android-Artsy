LOCAL_PATH := $(call my-dir)
OPENCV_PATH := /Users/jennifer/CS231m/android/OpenCV-2.4.8.2-Tegra-sdk

# Deep Belief
include $(CLEAR_VARS)
LOCAL_MODULE := jpcnn
LOCAL_SRC_FILES := DeepBelief/libjpcnn.so
include $(PREBUILT_SHARED_LIBRARY)

# Tegra optimized OpenCV.mk
include $(CLEAR_VARS)
OPENCV_INSTALL_MODULES:=on
include $(OPENCV_PATH)/sdk/native/jni/OpenCV-tegra3.mk

# Artsy
LOCAL_LDLIBS += -llog
LOCAL_SHARED_LIBRARIES  += jpcnn
LOCAL_MODULE    := Artsy
LOCAL_SRC_FILES := NativeLogging.cpp NativeCore.cpp  ar/ARApp.cpp ar/Augmentor.cpp ar/BinarySVM.cpp ar/Classifier.cpp ar/ConvNet.cpp ar/CornerDetection.cpp ar/KLTTracker.cpp ar/ORBTracker.cpp ar/PlaneTracker.cpp ar/ResourceLocator.cpp

include $(BUILD_SHARED_LIBRARY)
