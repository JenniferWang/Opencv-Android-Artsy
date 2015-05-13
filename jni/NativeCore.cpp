#include "NativeCore.hpp"
#include "ar/ARApp.hpp"
#include "ar/ResourceLocator.hpp"
#include <string>

static std::string ConvertString(JNIEnv* env, jstring js)
{
    const char* stringChars = env->GetStringUTFChars(js, 0);
    std::string s = std::string(stringChars);
    env->ReleaseStringUTFChars(js, stringChars);
    return s;
}

JNIEXPORT jlong JNICALL Java_edu_stanford_cvgl_artsy_CameraActivity_CreateNativeController
  (JNIEnv *, jobject)
{
	return (jlong)(new ar::ARApp);
}

JNIEXPORT void JNICALL Java_edu_stanford_cvgl_artsy_CameraActivity_DestroyNativeController
  (JNIEnv *, jobject, jlong addr_native_controller)
{
	delete (ar::ARApp*)(addr_native_controller);
}

JNIEXPORT void JNICALL Java_edu_stanford_cvgl_artsy_CameraActivity_HandleFrame
  (JNIEnv *, jobject, jlong addr_native_controller, jlong addr_rgba, jboolean is_init)
{
	ar::ARApp* app = (ar::ARApp*)(addr_native_controller);
	cv::Mat* frame = (cv::Mat*)(addr_rgba);
	if(is_init)
	{
		app->initialize(*frame);
	}
	else
	{
		app->process_frame(*frame);
	}
}

JNIEXPORT void JNICALL Java_edu_stanford_cvgl_artsy_CameraActivity_SetDataLocation
  (JNIEnv* env, jobject, jstring path)
{
	ar::SetResourceBasePath(ConvertString(env, path));
}
