//
// Created by kjm on 2020-06-23.
//

#include <jni.h>
#include "com_example_ndklibtest3_MainActivity.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


#if (!defined(WIN32) || !defined(_WIN64))
#define CV_FILLED -1
#define CV_RETR_EXTERNAL 0
#define CV_CHAIN_APPROX_SIMPLE 2

void FindCircle(cv::Mat image, int* x, int* y, int* radius, float resizeRatio);
void CircleGageToRect(cv::Mat out, cv::Mat* out2);
void RectImgThreshold(cv::Mat* inout);
void RemoveNonContinuousArea_Row(cv::Mat* inout);
void MaxFreq_Col(cv::Mat img, int* maxidxy, int* maxstartx, int* maxendx);
void FindIndicator(cv::Mat img, int *indicatorPosX);
void RemoveNonGrideLine_rect(cv::Mat *image);
void RemoveNonGrideLine_circle(cv::Mat *image);
void RemoveNumberArea(cv::Mat* image);
int FindMinMaxXY_contour(std::vector<cv::Point> contour, int *ix, int *iy, int *ax, int *ay);
void ImgThreshold_inv(cv::Mat* inout, int th);
void GetDetectArea(cv::Mat* image, int x, int y, int radius);
int FindMaxAreaContourIndex(std::vector<std::vector<cv::Point>> *contours);
void GetGrayArea(cv::Mat* image);
void RemoveNumberArea2(cv::Mat* image);
void AlignmentImg(cv::Mat in, cv::Mat* out);
void AlignmentImg_PostProcessing(cv::Mat* image);
void RemoveNonGrideLine_rect_morp(cv::Mat* image);
#endif


extern "C" {

JNIEXPORT float JNICALL Java_com_example_ndklibtest3_MainActivity_FindCircle
        //(JNIEnv *env, jobject instance, jlong matAddrInput, jint x, jint y, jint radius) {
        (JNIEnv *env, jobject instance, jlong matAddrInput, jobject param) {
    // 원 형태 검출
    Mat &matInput = *(Mat *) matAddrInput;

//    float ratio = 960. / matInput.cols;
    float ratio = 1024. / matInput.cols;

    Mat resizedimg;
    resize(matInput, resizedimg, Size(matInput.cols*ratio, matInput.rows*ratio),0, 0, INTER_AREA);

    int x, y, radius;
    FindCircle(resizedimg, &x, &y, &radius, ratio);

    jfieldID fid;
    jmethodID mid;
    jclass cls = env->GetObjectClass(param);
    fid = env->GetFieldID(cls, "x", "I");
    env->SetIntField(param, fid, x);
    fid = env->GetFieldID(cls, "y", "I");
    env->SetIntField(param, fid, y);
    fid = env->GetFieldID(cls, "radius", "I");
    env->SetIntField(param, fid, radius);

    return ratio;
}
}

extern "C" {

JNIEXPORT float JNICALL Java_com_example_ndklibtest3_MainActivity_AnalogGageIndicatorVal
        (JNIEnv *env, jobject instance, jlong matAddrInput, jint x, jint y, jint radius, jfloat resizeRatio, jlong matDebug) {

    Mat &inimg = *(Mat *) matAddrInput;
    Mat &outdebug = *(Mat *) matDebug;

    Mat out;
    Mat out1;
    Mat out2;
    Mat out3;

    cv::Mat in;
    resize(inimg, in, cv::Size(inimg.cols * resizeRatio, inimg.rows * resizeRatio), 0, 0,
           INTER_AREA);


    if (x < 50 * resizeRatio || y < 50 * resizeRatio || radius <= 50 * resizeRatio ||
        x > in.cols - 50 * resizeRatio || y > in.rows - 50 * resizeRatio) {
        return 0;
    }
    if (x - radius <= 0 || y - radius <= 0 || x - radius + radius * 2 >= in.cols ||
        y - radius + radius * 2 >= in.rows) {
        return 0;
    }

    Mat image;

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    image = in;
#else
    cv::cvtColor(in, image, COLOR_RGBA2BGR);
#endif

    cv::Rect rect_crop(x - radius, y - radius, radius * 2, radius * 2);
    image(rect_crop).copyTo(out);

    // 게이지 내부 영역 검출
    GetDetectArea(&out, x, y, radius);
#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("GetDetectArea", out);
    cv::waitKey(1);
#endif

    GetGrayArea(&out);
#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("GetGrayArea", out);
    cv::waitKey(1);
#endif


    RectImgThreshold(&out);

    cv::cvtColor(out, out1, COLOR_BGR2GRAY);
    cv::threshold(out1, out1, 1, 255, THRESH_BINARY);

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("filtering_th", out1);
    cv::waitKey(1);
#endif

    out2 = out1.clone();

    // 숫자부분 제거
    // contour 크기 및 area 기반 처리
    RemoveNumberArea2(&out2);
#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("RemoveNumberArea2", out2);
    cv::waitKey(1);
#endif

    cv::Mat out2_1;
    dilate(out2, out2_1, Mat::ones(3, 3, CV_8U));
#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("out2_dilate", out2_1);
    cv::waitKey(1);
#endif

//	RemoveNumberArea2(&out2_1);

    // 눈금 아닌 부분에 대한 제거
    RemoveNonGrideLine_circle(&out2_1);
#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("RemoveNonGrideLine_circle", out2_1);
    cv::waitKey(1);
#endif

    // 원 -> 직사각형 재구성
    CircleGageToRect(out2_1, &out3);

    cv::threshold(out3, out3, 1, 255, THRESH_BINARY);

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("out2_pre", out3);
    cv::waitKey(1);
#endif

    // 전체 이미지에서 세로로 가장 긴 위치 (눈금위치 계산)
    int indicatorPosX = 0;
//	Mat indicatorimg;
//	Mat out1_erode;
//	erode(out1, out1_erode, Mat::ones(3, 3, CV_8U));
//	dilate(out1_erode, out1_erode, Mat::ones(3, 3, CV_8U));
//	CircleGageToRect(out1_erode, &indicatorimg);
//	FindIndicator(indicatorimg, &indicatorPosX);
    FindIndicator(out3, &indicatorPosX);

    RemoveNonGrideLine_rect_morp(&out3);
    RemoveNonGrideLine_rect(&out3);

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("RemoveNonGrideLine", out3);
    cv::waitKey(1);
#endif

    Mat out4 = cv::Mat::zeros(out3.size(), CV_8UC1);

    AlignmentImg(out3, &out4);

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("AlignmentImg", out4);
    cv::waitKey(1);
#endif

    // 가로, 세로 기준으로 눈금영역 아닌 부분 제거
    AlignmentImg_PostProcessing(&out4);

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("AlignmentImg_post", out4);
    cv::waitKey(1);
#endif


    // col. frequency가 가장 높은값 계산
    int maxidxy = 0;
    int maxstartx = 0;
    int maxendx = 0;
    MaxFreq_Col(out4, &maxidxy, &maxstartx, &maxendx);

    // rect height margin : +-10
    int rect_y = maxidxy - 10;
    if (rect_y <= 0) rect_y = 0;
    if (rect_y >= out4.rows - 20) rect_y = out4.rows - 20;

    Rect rect(0, rect_y, out4.cols, 20);
    Mat out5 = cv::Mat::zeros(Size(out4.cols, 20), CV_8UC1);
    out4(rect).copyTo(out5);

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("AlignmentImg_rect", out5);
    cv::waitKey(1);
#endif


#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    // 눈금 위치 그리기
    Mat out6;
    cvtColor(out5, out6, CV_GRAY2RGB);

    // green line
    line(out6, Point(maxstartx, 10), Point(maxendx, 10), Scalar(0, 255, 0), 1);

    // red line
    uchar* p_out6 = (uchar*)out6.data;
    for (int j = 0; j < out6.rows; j++) {
//		p_out6[j*out5.cols * 3 + indicatorPosX * 3 + 2] = 255;
        for (int i = indicatorPosX - 1; i < indicatorPosX + 1; i++) {
            p_out6[j*out5.cols * 3 + i * 3 + 0] = 0;
            p_out6[j*out5.cols * 3 + i * 3 + 1] = 0;
            p_out6[j*out5.cols * 3 + i * 3 + 2] = 255;
        }
    }

    imshow("out4", out5);
    imshow("out5", out6);
    cv::waitKey(1);
#endif

    float ratio = (float) (indicatorPosX - maxstartx) / (maxendx - maxstartx);

    if (ratio < 0) ratio = 0;


#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    printf("ratio : %.3f\%%\n", ratio);

    char textbuf[255];

    sprintf(textbuf, "%.3f\%%", ratio);
    cv::putText(in, textbuf, Point(50, 50), 1, 3, Scalar(0, 0, 255), 2, 8, false);

    sprintf(textbuf, "%d psi", (int)(ratio * 3000));
    cv::putText(in, textbuf, Point(50, 100), 1, 3, Scalar(0, 0, 255), 2, 8, false);

    imshow("output", in);
    cv::waitKey(1);
#endif


    return ratio;
}
}

#if (defined(WIN32) || defined(_WIN64))
void CGage::GetGrayArea(cv::Mat* inout) {
#else
void GetGrayArea(cv::Mat* inout) {
#endif
    uchar* p_inout = (uchar*)inout->data;

    for (int p = 0; p < inout->rows; p++) {
        for (int q = 0; q < inout->cols; q++) {
            int r = p_inout[p*inout->cols * 3 + q * 3 + 2];
            int g = p_inout[p*inout->cols * 3 + q * 3 + 1];
            int b = p_inout[p*inout->cols * 3 + q * 3 + 0];

            if (abs(r - g) < 50 && abs(r - b) < 50 && abs(g - b) < 50) {

            }
            else {
                p_inout[p*inout->cols * 3 + q * 3 + 2] = 255;
                p_inout[p*inout->cols * 3 + q * 3 + 1] = 255;
                p_inout[p*inout->cols * 3 + q * 3 + 0] = 255;
            }
        }
    }
}

// 밝은 바탕에 어두운 눈금,게이지 형태인 경우
#if (defined(WIN32) || defined(_WIN64))
void CGage::GetDetectArea(cv::Mat* image, int x, int y, int radius) {
#else
void GetDetectArea(cv::Mat* image, int x, int y, int radius) {
#endif
    Mat gray;
    Mat thimg;
    cv::cvtColor(*image, gray, COLOR_BGR2GRAY);

    Mat circleimg = Mat::zeros(gray.size(), CV_8UC1);
    circle(circleimg, Point(gray.cols/2, gray.rows/2), radius, Scalar(255), CV_FILLED);

    bitwise_and(gray, circleimg, gray);

    cv::threshold(gray, thimg, 128, 255, THRESH_BINARY);
//	imshow("thimg", thimg);
//	cv::waitKey(1);

    vector<vector<Point> > contours_out;
    vector<Vec4i> hierarchy;
    cv::findContours(thimg.clone(), contours_out, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    int maxidx = FindMaxAreaContourIndex(&contours_out);

    Mat maxcontourimg = Mat::zeros(gray.size(), CV_8UC1);

    drawContours(maxcontourimg, contours_out, maxidx, Scalar(255), CV_FILLED);
//	imshow("maxcontourimg", maxcontourimg);
//	cv::waitKey(1);

    Mat out = Mat::zeros(gray.size(), CV_8UC3);
    bitwise_not(out, out);

    image->copyTo(out, maxcontourimg);
//	imshow("valid_area only", out);
//	cv::waitKey(1);

    *image = out;
}

#if (defined(WIN32) || defined(_WIN64))
void CGage::AlignmentImg(cv::Mat in, cv::Mat* out) {
#else
void AlignmentImg(cv::Mat in, cv::Mat* out) {
#endif
    int width = in.cols;
    int height = in.rows;

    uchar* p_in = (uchar*)in.data;
    uchar* p_out = (uchar*)out->data;

    for (int q = 0; q < in.cols; q++) {
        int starty = 0;
        for (int p = 0; p < in.rows/2; p++) {
            int val = p_in[p*in.cols + q];
            if (val > 0) {
                starty = p;
                break;
            }
        }
        // 이미지 shift
        for (int p = starty; p < in.rows/2; p++) {
            if (p_in[p*in.cols + q] > 0) {
                p_out[(p - starty)*in.cols + q] = p_in[p*in.cols + q];
            }
        }
    }
}

#if (defined(WIN32) || defined(_WIN64))
void CGage::AlignmentImg_PostProcessing(cv::Mat* in) {
#else
void AlignmentImg_PostProcessing(cv::Mat* in) {
#endif
    int width = in->cols;
    int height = in->rows;

    uchar* p_in = (uchar*)in->data;

    // 가로 확인
    for (int p = 0; p < in->rows / 2; p++) {
        int wval = 0;

        for (int i = 0; i < in->cols; i++) {
            int val = p_in[p*in->cols + i];
            if (val > 0) {
                wval++;
                // 가로로 연속된 255가 많다면 한줄 제거 가능
                if (wval > in->cols*0.05) {
                    for (int k = 0; k < in->cols; k++) {
                        p_in[p*in->cols + k] = 0;
                    }
                    break;
                }
            }
            else {
                wval = 0;
            }
        }

    }

//	imshow("outimage_post_cols", *in);
//	cv::waitKey(0);

    // 세로
    int startysum = 0;
    int cnt = 0;
    for (int q = 0; q < in->cols; q++) {
        int wval = 0;
        for (int p = 0; p < in->rows / 2; p++) {
            int val = p_in[p*in->cols + q];
            if (val == 0) {
                if (p != 0 && wval>0) {
                    startysum += p;
                    cnt++;
                    break;
                }
            }
            else {
                wval++;
            }
        }
    }
    int avg = (float)startysum / cnt;

    for (int q = 0; q < in->cols; q++) {
        int rowcnt = 0;
        int cnt = 0;
        int wcnt = 0;
        for (int p = 0; p < in->rows / 2; p++) {
            int val = p_in[p*in->cols + q];
            if (val > 0){
                wcnt++;
            }
        }

        if (wcnt < avg*0.5) {
            for (int p = 0; p < in->rows / 2; p++) {
                p_in[p*in->cols + q] = 0;
            }
        }

    }

//	imshow("outimage_post_rows", *in);
//	cv::waitKey(0);

}

#if (defined(WIN32) || defined(_WIN64))
void CGage::RemoveNumberArea(cv::Mat* image) {
#else
void RemoveNumberArea(cv::Mat* image) {
#endif

    int width = image->cols;
    int height = image->rows;

    Mat numAreaImg = cv::Mat::zeros(image->size(), CV_8UC1);
    Mat mask;
    erode(image->clone(), mask, Mat::ones(3, 3, CV_8U));
    dilate(mask, mask, Mat::ones(3, 3, CV_8U));

    dilate(mask, mask, Mat::ones(9, 9, CV_8U));

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("mask", mask);
	cv::waitKey(1);
#endif

    vector<vector<Point> > contours_out;
    vector<Vec4i> hierarchy;
    cv::findContours(mask.clone(), contours_out, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    mask = cv::Mat::zeros(image->size(), CV_8UC1);

    for (int i = 0; i < contours_out.size(); i++) {
        int maxx, maxy, minx, miny;
        FindMinMaxXY_contour(contours_out[i], &minx, &miny, &maxx, &maxy);
        int contourW = maxx - minx;
        int contourH = maxy - miny;
        if (contourW > 10 && contourW < width*0.6 && contourH>10 && contourH<height*0.5) {
            drawContours(mask, contours_out, i, Scalar(255), CV_FILLED);
        }

    }

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("mask2", mask);
	cv::waitKey(1);
#endif

    Mat imgresult = *image - mask;

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("imgresult", imgresult);
	cv::waitKey(1);
#endif

    *image = imgresult;
}

#if (defined(WIN32) || defined(_WIN64))
void CGage::RemoveNumberArea2(cv::Mat* image) {
#else
void RemoveNumberArea2(cv::Mat* image) {
#endif
    int width = image->cols;
    int height = image->rows;

    vector<vector<Point> > contours_out;
    vector<Vec4i> hierarchy;
    cv::findContours(image->clone(), contours_out, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    Mat numAreaImg = cv::Mat::zeros(image->size(), CV_8UC1);

    for (int i = 0; i < contours_out.size(); i++) {
        int maxx, maxy, minx, miny;
        FindMinMaxXY_contour(contours_out[i], &minx, &miny, &maxx, &maxy);
        int contourW = maxx - minx;
        int contourH = maxy - miny;

        if (contourW < width*0.15 && contourH < height*0.15) {
            int area = (int)contourArea(contours_out[i], false);
            // 글자부분만 그리기
            if (area > width*height*0.001) {
                drawContours(numAreaImg, contours_out, i, Scalar(255), CV_FILLED);
            }
        }

    }
    // 글자부분 빼기
    *image -= numAreaImg;
}


#if (defined(WIN32) || defined(_WIN64))
void CGage::RemoveNonGrideLine_rect(cv::Mat* image) {
#else
void RemoveNonGrideLine_rect(cv::Mat* image) {
#endif

    uchar* p_image = (uchar*)image->data;
    int width = image->cols;
    int height = image->rows;

    int* buf;
    buf = new int[width];

    int h_masksize = image->rows*0.2;
    int search_h = height - h_masksize;
    float blackCntTh = 0.1;

    for (int i = 0; i < search_h ; i++) {
        for (int j = 0; j < width; j++) {
            buf[j] = 0;
        }
        // row에 대하여 분석 (하나라도 white가 있다면 255)
        for (int j = 0; j < width; j++) {
            int temp = 0;
            for (int k = i; k < i+h_masksize; k++) {
                int val = p_image[k*width + j];
                if (val > 0 ) {
                    temp = 255;
                    break;
                }
            }
            buf[j] = temp;
        }
        // 주기적이지 않다면 전체를 제거
        int candidate = 0;
        for (int j = 0; j < width; j++) {
            int blackcnt = 0;
            // 검정색 영역이 넓게 분포한다면 제거 대상
            int k = j;
            for ( ; k < width; k++) {
                if (buf[k] == 0) {
                    blackcnt++;
                }
                else {
                    break;
                }
            }

            if (blackcnt > width * blackCntTh) {
                candidate++;
            }
            if (blackcnt > width*0.3) {
                candidate += 10;
            }
            j = k;
        }

        // candidate 를 보고 판단
        if (candidate > 2) {
            for (int p = i; p < i + h_masksize; p++) { // row
                for (int j = 0; j < width; j++) {
                    p_image[p*width + j] = 0;
                }
            }
            i += h_masksize-1;
        }
    }
    // 아래부분 제거
    for (int i = height*2/3; i < height; i++) {
        for (int j = 0; j < width; j++) {
            p_image[i*width + j] = 0;
        }
    }

}

#if (defined(WIN32) || defined(_WIN64))
void CGage::RemoveNonGrideLine_rect_morp(cv::Mat* image) {
#else
void RemoveNonGrideLine_rect_morp(cv::Mat* image) {
#endif
    Mat dilateImg;
    dilate(*image, dilateImg, Mat::ones(15, 15, CV_8U));

    Mat mask = cv::Mat::zeros(image->size(), CV_8UC1);

    vector<vector<Point> > contours_out;
    vector<Vec4i> hierarchy;
    cv::findContours(dilateImg.clone(), contours_out, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours_out.size(); i++) {
        int maxx, maxy, minx, miny;
        FindMinMaxXY_contour(contours_out[i], &minx, &miny, &maxx, &maxy);
        int contourW = maxx - minx;
        if (contourW > image->cols / 2) {
            drawContours(mask, contours_out, i, Scalar(255), CV_FILLED);
        }
    }

    bitwise_and(*image, mask, *image);
}

#if (defined(WIN32) || defined(_WIN64))
void CGage::RemoveNonGrideLine_circle(cv::Mat* image) {
#else
void RemoveNonGrideLine_circle(cv::Mat* image) {
#endif

    // 원 형태의 이미지에서 눈금이 아닌 부분 제거
    // contour의 넓이 기준
    vector<vector<Point> > contours_out;
    vector<Vec4i> hierarchy;
    cv::findContours(image->clone(), contours_out, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // 눈금이 분리되어 개수가 많은 경우에 한하여 수행
    if (contours_out.size() > 80) {

        int sum = 0;
        for (int i = 0; i < contours_out.size(); i++) {
            int area = (int)contourArea(contours_out[i], false);
            sum += area;
        }

        int avgarea = sum / contours_out.size();

        *image = cv::Mat::zeros(image->size(), CV_8UC1);

        for (int i = 0; i < contours_out.size(); i++) {
            int area = (int)contourArea(contours_out[i], false);
            if (area > avgarea*0.3 && area < avgarea * 2) {
                drawContours(*image, contours_out, i, Scalar(255), CV_FILLED);
            }
        }

        // 중심점에서 가까운 것은 포함시켜줌
        std::vector<int> centerdiff;
        int x, y;
        x = y = 0;
        for (int i = 0; i < contours_out.size(); i++) {
            int minval = 99999;
            int idx = 0;
            for (int j = 0; j < contours_out[i].size(); j++) {
                int val = sqrt(pow(image->cols / 2 - contours_out[i][j].x, 2) + pow(image->rows / 2 - contours_out[i][j].y, 2));
                if (minval > val) {
                    minval = val;
                }
            }
            centerdiff.push_back(minval);
        }

        for (int i = 0; i < centerdiff.size(); i++) {
            if (centerdiff[i] < image->cols / 2 * 0.3) {
                drawContours(*image, contours_out, i, Scalar(255), CV_FILLED);
            }
        }



        // 바늘영역은 포함 시켜야 함
        // center가 중심점과 가장 가까운 컨투어는 그려줌
//		std::vector<int> centerdiff;
//		int x, y;
//		x = y = 0;
//		for (int i = 0; i < contours_out.size(); i++) {
//			findContourCenter(contours_out[i], &x, &y);
//			int diff = sqrt(pow(image->cols/2 - x, 2) + pow(image->rows/2 - y, 2));
//			centerdiff.push_back(diff);
//		}
//		int mindiff = 99999;
//		int mindiff_idx = 0;
//		for (int i = 0; i < centerdiff.size(); i++) {
//			if (mindiff > centerdiff[i]) {
//				mindiff = centerdiff[i];
//				mindiff_idx = i;
//			}
//		}
//		drawContours(*image, contours_out, mindiff_idx, Scalar(255), CV_FILLED);
    }
}

#if (defined(WIN32) || defined(_WIN64))
void CGage::findContourCenter(std::vector<cv::Point> contours, int* cx, int* cy) {
#else
void findContourCenter(std::vector<cv::Point> contours, int* cx, int* cy) {
#endif
    int x, y;
    x = y = 0;

    for (int i = 0; i < contours.size(); i++) {
        x += contours[i].x;
        y += contours[i].y;
    }

    *cx = x / contours.size();
    *cy = y / contours.size();
}


#if (defined(WIN32) || defined(_WIN64))
void CGage::FindIndicator(cv::Mat img, int *indicatorPosX) {
#else
void FindIndicator(cv::Mat img, int *indicatorPosX) {
#endif
    int maxcnt = 0;
    uchar* p_img = (uchar*)img.data;


//	for (int q = 0; q < img.cols - 1; q++) {
//		int cnt = 0;
//		for (int p = 0; p < img.rows/2; p++) {
//			if (p_img[p*img.cols + q] == 255) {
//				cnt++;
//			}
//		}
//		if (maxcnt < cnt) {
//			maxcnt = cnt;
//			*indicatorPosX = q;
//		}
//	}

    // 연속된 개수가 가장 많은 영역 검출
    for (int q = 0; q < img.cols - 1; q++) {
        int cnt = 0;
        for (int p = 0; p < img.rows; p++) {
            if (p_img[p*img.cols + q] == 255) {
                cnt++;
            }
        }
        if (maxcnt < cnt) {
            maxcnt = cnt;
            *indicatorPosX = q;
        }
    }

}

#if (defined(WIN32) || defined(_WIN64))
void CGage::MaxFreq_Col(cv::Mat img, int* maxidxy, int* maxstartx, int* maxendx) {
#else
void MaxFreq_Col(cv::Mat img, int* maxidxy, int* maxstartx, int* maxendx) {
#endif
    *maxidxy = *maxstartx = *maxendx=0;

    uchar* p_img = (uchar*)img.data;
    int maxcnt = 0;
    for (int p = 0; p < img.rows; p++) {
        int cnt = 0;
        int startx, endx;
        startx = 0;
        endx = 0;
        for (int q = 0; q < img.cols - 1; q++) {
            if (p_img[p*img.cols + q] == 0 && p_img[p*img.cols + q + 1] == 255) {
                cnt++;
                if (startx == 0) {
                    startx = q+1;
                }
            }

            if (p_img[p*img.cols + q] == 255 && p_img[p*img.cols + q + 1] == 0) {
                cnt++;
                endx = q;
            }

        }
        if (maxcnt < cnt) {
            maxcnt = cnt;
            *maxidxy = p;
            // 두께 고려하여 재계산
            int thicknessCnt = 0;
            for (int q = startx; q < startx+10; q++) {
                if (p_img[p*img.cols + q] > 0) {
                    thicknessCnt++;
                }
                else {
                    break;
                }

            }
            *maxstartx = startx + thicknessCnt/2;

            thicknessCnt = 0;
            for (int q = endx; q > endx-10; q--) {
                if (p_img[p*img.cols + q] > 0) {
                    thicknessCnt++;
                }
                else {
                    break;
                }

            }

            *maxendx = endx - thicknessCnt / 2;
        }
    }
}

#if (defined(WIN32) || defined(_WIN64))
void CGage::ImgThreshold_inv(cv::Mat* inout, int th) {
#else
void ImgThreshold_inv(cv::Mat* inout, int th) {
#endif
    uchar* p_inout = (uchar*)inout->data;

    for (int p = 0; p < inout->rows; p++) {
        for (int q = 0; q < inout->cols; q++) {

            if (p_inout[p*inout->cols + q]<th) {
                p_inout[p*inout->cols + q] = 255;
            }
            else {
                p_inout[p*inout->cols + q] = 0;
            }
        }
    }

}

// 게이지 종류에 따라 추후 수정 필요
// 현재 :	r > 128
//			g-r, g-b, r-b 차이 15 이상
// 제거영역 white로 수정
#if (defined(WIN32) || defined(_WIN64))
void CGage::RectImgThreshold(cv::Mat* inout) {
#else
void RectImgThreshold(cv::Mat* inout) {
#endif
    uchar* p_inout = (uchar*)inout->data;

    int hist_r[256];
    int hist_g[256];
    int hist_b[256];
    for (int i = 0; i < 256; i++) {
        hist_r[i] = 0;
        hist_g[i] = 0;
        hist_b[i] = 0;
    }

    for (int p = 0; p < inout->rows; p++) {
        for (int q = 0; q < inout->cols; q++) {
            int r = p_inout[p*inout->cols*3 + q*3+2];
            int g = p_inout[p*inout->cols * 3 + q * 3 + 1];
            int b = p_inout[p*inout->cols * 3 + q * 3 + 0];

            if (r < 128 && g < 128 && b < 128) {
                if (abs(r - g) < 50 && abs(r - b) < 50 && abs(g - b) < 50) {
                    hist_r[r]++;
                    hist_g[g]++;
                    hist_b[b]++;
                }
            }
        }
    }

    int maxr = 0;
    int maxridx = 0;
    for (int p = 0; p < 256; p++) {
        if (maxr < hist_r[p]) {
            maxr = hist_r[p];
            maxridx = p;
        }
    }
    int maxg = 0;
    int maxgidx = 0;
    for (int p = 0; p < 256; p++) {
        if (maxg < hist_g[p]) {
            maxg = hist_g[p];
            maxgidx = p;
        }
    }
    int maxb = 0;
    int maxbidx = 0;
    for (int p = 0; p < 256; p++) {
        if (maxb < hist_b[p]) {
            maxb = hist_b[p];
            maxbidx = p;
        }
    }

    int margin = 90;// 70;

    for (int p = 0; p < inout->rows; p++) {
        for (int q = 0; q < inout->cols; q++) {
            int r = p_inout[p*inout->cols * 3 + q * 3 + 2];
            int g = p_inout[p*inout->cols * 3 + q * 3 + 1];
            int b = p_inout[p*inout->cols * 3 + q * 3 + 0];

            if (r > maxridx - margin && r<maxridx + margin && g>maxgidx - margin && g<maxgidx + margin && b>maxbidx - margin && b < maxbidx + margin) {
                p_inout[p*inout->cols * 3 + q * 3 + 2] = 255;
                p_inout[p*inout->cols * 3 + q * 3 + 1] = 255;
                p_inout[p*inout->cols * 3 + q * 3 + 0] = 255;
            }
            else {
                p_inout[p*inout->cols * 3 + q * 3 + 2] = 0;
                p_inout[p*inout->cols * 3 + q * 3 + 1] = 0;
                p_inout[p*inout->cols * 3 + q * 3 + 0] = 0;

            }

        }
    }
}

#if (defined(WIN32) || defined(_WIN64))
void CGage::RemoveNonContinuousArea_Row(cv::Mat* inout) {
#else
void RemoveNonContinuousArea_Row(cv::Mat* inout) {
#endif
    uchar* p_inout = (uchar*)inout->data;

    // 세로쪽 분석하여 연속된 값이 존재하지 않는다면 제거
    // 게이지의 눈금이 외곽에 위치한 경우에 한함.
    if (0) { // color
        for (int q = 0; q < inout->cols; q++) {
            int cnt = 0;

            for (int p = 0; p < inout->rows / 2; p++) { // 세로
                int r = p_inout[(p)*inout->cols * 3 + q * 3 + 2];
                int g = p_inout[(p)*inout->cols * 3 + q * 3 + 1];
                int b = p_inout[(p)*inout->cols * 3 + q * 3 + 0];

                if (r > 128 && g > 128 && b > 128) {
                    cnt++;
                }
            }

            if (cnt < 20) {
                for (int i = 0; i < inout->rows / 2; i++) {
                    p_inout[(i)*inout->cols * 3 + q * 3 + 2] = 0;
                    p_inout[(i)*inout->cols * 3 + q * 3 + 1] = 0;
                    p_inout[(i)*inout->cols * 3 + q * 3 + 0] = 0;
                }
            }
        }
    }
    else {//gray
        for (int q = 0; q < inout->cols; q++) {
            int cnt = 0;

            for (int p = 0; p < inout->rows / 2; p++) { // 세로
                int val = p_inout[(p)*inout->cols + q ];
                if (val > 128) {
                    cnt++;
                }
            }

            if (cnt < 20) {
                for (int i = 0; i < inout->rows / 2; i++) {
                    p_inout[(i)*inout->cols + q ] = 0;
                }
            }
        }
    }
}


// 원 형태의 검출 영역을 사각형 형태로 변환
// 입력 영상은 정사각형 (원을 포함하는 최소크기)
#if (defined(WIN32) || defined(_WIN64))
void CGage::CircleGageToRect(cv::Mat out, cv::Mat* out2) {
#else
void CircleGageToRect(cv::Mat out, cv::Mat* out2) {
#endif
    int anglestep = 1;
    int w = 1;
    int h = out.rows/2;
    int gatherwidth = 360/anglestep * w;
    Mat gatherimg = cv::Mat::zeros(Size(gatherwidth, h), CV_8UC1);
    uchar* p_gatherimg = (uchar*)gatherimg.data;

    int width = out.cols;
    int height = out.rows;

    int cnt = 0;
    for (int i = 0; i <= 360; i = i + anglestep) {
        Mat matrix = getRotationMatrix2D(cv::Point2f(width / 2, height / 2), i, 1);
        Mat temp = cv::Mat::zeros(out.size(), CV_8UC1);
        cv::warpAffine(out, temp, matrix, out.size());

        Rect rect(out.cols / 2 - w/2, out.rows / 2, w, out.rows / 2-1);
        Mat subimage;

        temp(rect).copyTo(subimage);

        cv::flip(subimage, subimage, 0);

        uchar* p_subimage = (uchar*)subimage.data;

        for (int p = 0; p < subimage.rows; p++) {
            for (int q = 0; q < subimage.cols; q++) {
                if (1) {
                    p_gatherimg[p*gatherwidth + (q + cnt) + 0] = p_subimage[p*subimage.cols + q ];
                }
                else {
                    p_gatherimg[p*gatherwidth * 3 + (q + cnt) * 3 + 0] = p_subimage[p*subimage.cols * 3 + q * 3];
                    p_gatherimg[p*gatherwidth * 3 + (q + cnt) * 3 + 1] = p_subimage[p*subimage.cols * 3 + q * 3 + 1];
                    p_gatherimg[p*gatherwidth * 3 + (q + cnt) * 3 + 2] = p_subimage[p*subimage.cols * 3 + q * 3 + 2];
                }
            }
        }
        cnt = cnt + w;
    }

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("gatherimage", gatherimg);
	cv::waitKey(1);
#endif

    *out2 = gatherimg.clone();
}


// 입력 이미지에서 원형태를 검출하여 이미지 cropping 후 리턴
// resize 상태에서의 x, y, radius 계산됨.
#if (defined(WIN32) || defined(_WIN64))
void CGage::FindCircle(cv::Mat image, int* x, int* y, int* radius, float resizeRatio) {
#else
void FindCircle(cv::Mat image, int* x, int* y, int* radius, float resizeRatio) {
#endif
    Mat resizedimg = image;

    Mat gray;
    Mat gray_blur;
    Mat temp = resizedimg.clone();
#if (defined(WIN32) || defined(_WIN64))
    cvtColor(resizedimg, gray, COLOR_BGR2GRAY);
#else
    cvtColor(resizedimg, gray, COLOR_RGBA2GRAY);
#endif

    blur(gray, gray_blur, Size(21, 21));


#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    imshow("gray", gray_blur);
	waitKey(1);
#endif

    vector<Vec3f> circles;
    //cv2.HoughCircles(image, circles, method, dp, minDist, param1, param2, minRadius, maxRadius)
    //cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,	param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
    //			minDist : 검출한 원의 중심과의 최소거리. 값이 작으면 원이 아닌 것들도 검출이 되고, 너무 크면 원을 놓칠 수 있음.
    //			param1 : canny edge parameter
    //			param2 : 값이 작으면 오류 높고, 크면 검출율 낮아짐

//	HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 100, 100, gray.rows/8, gray.rows/2);
    HoughCircles(gray_blur, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 50, 50, gray.rows / 4, gray.rows*2/3);

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center = Point(c[0], c[1]);
		// circle center
		circle(temp, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
		// circle outline
		int radius = c[2];
		circle(temp, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
	}
	imshow("detected circles", temp);
	waitKey(1);
#endif

    if (circles.size() == 1) {
        Rect rt(circles[0][0] - circles[0][2], circles[0][1] - circles[0][2], circles[0][2] * 2, circles[0][2] * 2);
        *x = circles[0][0];
        *y = circles[0][1];
        *radius = circles[0][2];
    }
    else if (circles.size() > 1) {
        HoughCircles(gray_blur, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 50, 100, gray.rows / 4, gray.rows * 2 / 3);

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
        for (size_t i = 0; i < circles.size(); i++)
		{
			Vec3i c = circles[i];
			Point center = Point(c[0], c[1]);
			// circle center
			circle(temp, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
			// circle outline
			int radius = c[2];
			circle(temp, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
		}
		imshow("detected circles", temp);
		waitKey(1);
#endif

        if (circles.size() == 1) {
            Rect rt(circles[0][0] - circles[0][2], circles[0][1] - circles[0][2], circles[0][2] * 2, circles[0][2] * 2);
            *x = circles[0][0];
            *y = circles[0][1];
            *radius = circles[0][2];
        }
    }
    else if(circles.size() == 0) {
        // 미검출 시 median blur 수행 후 다시한번 더 수행
        medianBlur(gray, gray_blur, 21);
        HoughCircles(gray_blur, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 50, 50, gray.rows / 4, gray.rows * 2 / 3);
#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
        for (size_t i = 0; i < circles.size(); i++)
		{
			Vec3i c = circles[i];
			Point center = Point(c[0], c[1]);
			// circle center
			circle(temp, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
			// circle outline
			int radius = c[2];
			circle(temp, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
		}
		imshow("detected circles", temp);
		waitKey(1);
#endif
        if (circles.size() == 1) {
            Rect rt(circles[0][0] - circles[0][2], circles[0][1] - circles[0][2], circles[0][2] * 2, circles[0][2] * 2);
            //		image(rt).copyTo(*outimg);	// ROI 복사 시 주의
            *x = circles[0][0];
            *y = circles[0][1];
            *radius = circles[0][2];
        }
        else {
            *x = *y = *radius = 0;
        }
    }

#if (defined(WIN32) || defined(_WIN64)) && defined(DEBUG_CODE)
    //	imshow("cropimg", *outimg);
//	waitKey(1);
#endif
}

#if (defined(WIN32) || defined(_WIN64))
int CGage::FindMinMaxXY_contour(std::vector<cv::Point> contour, int *ix, int *iy, int *ax, int *ay) {
#else
int FindMinMaxXY_contour(std::vector<cv::Point> contour, int *ix, int *iy, int *ax, int *ay) {
#endif
    int i;
    *ix = *iy = 9999;
    *ax = *ay = 0;

    for (i = 0; i < contour.size(); i++) {
        int x = contour[i].x;
        int y = contour[i].y;
        if (*ax < x) {
            *ax = x;
        }
        if (*ix > x) {
            *ix = x;
        }

        if (*ay < y) {
            *ay = y;
        }
        if (*iy > y) {
            *iy = y;
        }
    }


    return 1;
}

#if (defined(WIN32) || defined(_WIN64))
int CGage::FindMaxAreaContourIndex(std::vector<std::vector<cv::Point>> *contours)
#else
int FindMaxAreaContourIndex(std::vector<std::vector<cv::Point>> *contours)
#endif
{
    if (!contours)
        return -1;

    if (contours->size() <= 0) {
        return -1;
    }

    double largest_area = -1;
    int largest_contour_index = -1;
    for (int i = 0; i<contours->size(); i++)
    {
        double a = contourArea((*contours)[i], false);
        if (a > largest_area)
        {
            largest_area = a;
            largest_contour_index = i;
        }
    }
    return largest_contour_index;

}