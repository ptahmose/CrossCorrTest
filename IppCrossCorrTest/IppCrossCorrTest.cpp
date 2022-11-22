// CMakeProject1.cpp : Defines the entry point for the application.
//

#include "IppCrossCorrTest.h"

#include <ipp.h>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

static void TestIppROIFull(int sourceWidth, int sourceHeight, int templateWidth, int templateHeight);
static void TestIppROIValid(int sourceWidth, int sourceHeight, int templateWidth, int templateHeight);
static void TestOpenCV(int sourceWidth, int sourceHeight, int templateWidth, int templateHeight);

int main()
{
    TestIppROIValid(32000, 2000, 10000, 1000);

    TestOpenCV(3200, 400, 2000, 200);
    TestOpenCV(32000, 6000, 10000, 3000);
    TestIppROIFull(32000, 2000, 10000, 1000);
    return 0;
}

void TestIppROIFull(int sourceWidth, int sourceHeight, int templateWidth, int templateHeight)
{
    int sourceStride;
    unique_ptr< Ipp32f, void(*)(void*) > upPtrSource{ ippiMalloc_32f_C1(sourceWidth, sourceHeight, &sourceStride), ippFree };
    ippiImageJaehne_32f_C1R(upPtrSource.get(), sourceStride, IppiSize{ sourceWidth, sourceHeight });

    int templateStride;
    unique_ptr< Ipp32f, void(*)(void*) > upPtrTemplate{ ippiMalloc_32f_C1(templateWidth, templateHeight, &templateStride), ippFree };
    ippiImageJaehne_32f_C1R(upPtrTemplate.get(), templateStride, IppiSize{ templateWidth, templateHeight });

    int destinationStride;
    int destinationWidth = sourceWidth + templateWidth - 1;
    int destinationHeight = sourceHeight + templateHeight - 1;
    unique_ptr< Ipp32f, void(*)(void*) > upPtrDestination{ ippiMalloc_32f_C1(destinationWidth, destinationHeight, &destinationStride), ippFree };
    ippiSet_32f_C1R(0.f, upPtrDestination.get(), destinationStride, IppiSize{ destinationWidth, destinationHeight });

    int bufferSize;
    IppStatus status = ippiCrossCorrNormGetBufferSize(
        IppiSize{ sourceWidth, sourceHeight },
        IppiSize{ templateWidth, templateHeight },
        ippAlgAuto | ippiNorm | ippiROIFull,
        &bufferSize);

    unique_ptr<Ipp8u, void(*)(void*)> upTempBuffer{ (Ipp8u*)malloc(bufferSize), free };

    // https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/image-statistics-functions/image-proximity-measures/crosscorrnorm-1.html
    status = ippiCrossCorrNorm_32f_C1R(
        upPtrSource.get(),
        sourceStride,
        IppiSize{ sourceWidth, sourceHeight },
        upPtrTemplate.get(),
        templateStride,
        IppiSize{ templateWidth, templateHeight },
        upPtrDestination.get(),
        destinationStride,
        ippAlgAuto | ippiNorm | ippiROIFull,
        upTempBuffer.get());
}

void TestIppROIValid(int sourceWidth, int sourceHeight, int templateWidth, int templateHeight)
{
    int sourceStride;
    unique_ptr< Ipp32f, void(*)(void*) > upPtrSource{ ippiMalloc_32f_C1(sourceWidth, sourceHeight, &sourceStride), ippFree };
    ippiImageJaehne_32f_C1R(upPtrSource.get(), sourceStride, IppiSize{ sourceWidth, sourceHeight });

    int templateStride;
    unique_ptr< Ipp32f, void(*)(void*) > upPtrTemplate{ ippiMalloc_32f_C1(templateWidth, templateHeight, &templateStride), ippFree };
    ippiImageJaehne_32f_C1R(upPtrTemplate.get(), templateStride, IppiSize{ templateWidth, templateHeight });

    int destinationStride;
    int destinationWidth = sourceWidth - templateWidth + 1;
    int destinationHeight = sourceHeight - templateHeight + 1;
    unique_ptr< Ipp32f, void(*)(void*) > upPtrDestination{ ippiMalloc_32f_C1(destinationWidth, destinationHeight, &destinationStride), ippFree };
    ippiSet_32f_C1R(0.f, upPtrDestination.get(), destinationStride, IppiSize{ destinationWidth, destinationHeight });

    int bufferSize;
    IppStatus status = ippiCrossCorrNormGetBufferSize(
        IppiSize{ sourceWidth, sourceHeight },
        IppiSize{ templateWidth, templateHeight },
        ippAlgAuto | ippiNorm | ippiROIValid,
        &bufferSize);

    unique_ptr<Ipp8u, void(*)(void*)> upTempBuffer{ (Ipp8u*)malloc(bufferSize), free };

    status = ippiCrossCorrNorm_32f_C1R(
        upPtrSource.get(),
        sourceStride,
        IppiSize{ sourceWidth, sourceHeight },
        upPtrTemplate.get(),
        templateStride,
        IppiSize{ templateWidth, templateHeight },
        upPtrDestination.get(),
        destinationStride,
        ippAlgAuto | ippiNorm | ippiROIValid,
        upTempBuffer.get());
}

void TestOpenCV(int sourceWidth, int sourceHeight, int templateWidth, int templateHeight)
{
    // https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
    // https://docs.opencv.org/3.4/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be
    int sourceStride;
    unique_ptr< Ipp32f, void(*)(void*) > upPtrSource{ ippiMalloc_32f_C1(sourceWidth, sourceHeight, &sourceStride), ippFree };
    ippiImageJaehne_32f_C1R(upPtrSource.get(), sourceStride, IppiSize{ sourceWidth, sourceHeight });

    int templateStride;
    unique_ptr< Ipp32f, void(*)(void*) > upPtrTemplate{ ippiMalloc_32f_C1(templateWidth, templateHeight, &templateStride), ippFree };
    ippiImageJaehne_32f_C1R(upPtrTemplate.get(), templateStride, IppiSize{ templateWidth, templateHeight });

    int destinationStride;
    int destinationWidth = sourceWidth + templateWidth - 1;
    int destinationHeight = sourceHeight + templateHeight + 1;
    unique_ptr< Ipp32f, void(*)(void*) > upPtrDestination{ ippiMalloc_32f_C1(destinationWidth, destinationHeight, &destinationStride), ippFree };
    ippiSet_32f_C1R(0.f, upPtrDestination.get(), destinationStride, IppiSize{ destinationWidth, destinationHeight });

    Mat sourceMat{ sourceHeight,sourceWidth,CV_32F,upPtrSource.get(), (size_t)sourceStride };
    Mat templateMat{ templateHeight,templateWidth,CV_32F,upPtrTemplate.get(), (size_t)templateStride };
    Mat destinationMat{ destinationHeight,destinationWidth,CV_32F,upPtrDestination.get(), (size_t)destinationStride };

    matchTemplate(sourceMat, templateMat, destinationMat, TM_CCORR_NORMED);
}