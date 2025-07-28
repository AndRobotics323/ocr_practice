#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// 안전한 OpenCV grayscale → Leptonica Pix 변환 함수
Pix* mat8ToPix(const cv::Mat& mat) {
    int width = mat.cols;
    int height = mat.rows;

    Pix* pix = pixCreate(width, height, 8);  // 8bit grayscale
    l_uint32* data = pixGetData(pix);
    int wpl = pixGetWpl(pix);  // word per line

    for (int y = 0; y < height; ++y) {
        l_uint32* line = data + y * wpl;
        const uchar* row = mat.ptr(y);
        for (int x = 0; x < width; ++x) {
            SET_DATA_BYTE(line, x, row[x]);  // Leptonica macro
        }
    }
    return pix;
}

int main() {
    // 1. 이미지 불러오기
    cv::Mat img = cv::imread("sample.png", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다.\n";
        return 1;
    }

    // 2. 그레이스케일 변환
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 3. 이진화 (선택적 전처리, 명도 임계값 자동 계산)
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);


    cv::imshow("Binary Image", gray);
    cv::waitKey(0); 

    // 4. Tesseract 엔진 초기화
    tesseract::TessBaseAPI tess;
    if (tess.Init(NULL, "kor+eng", tesseract::OEM_LSTM_ONLY)) {
        std::cerr << "Tesseract 초기화 실패\n";
        return 1;
    }

    
    // 5. OpenCV Mat → Leptonica Pix 변환
    Pix* pix = mat8ToPix(gray);

    // 6. 이미지 설정 및 OCR 수행
    tess.SetImage(pix);
    char* out = tess.GetUTF8Text();
    std::cout << "OCR 결과:\n" << out << std::endl;

    // 7. 리소스 정리
    delete[] out;      // OCR 텍스트 메모리 해제
    tess.End();        // Tesseract 내부 정리
    pixDestroy(&pix);  // Leptonica 이미지 메모리 해제

    return 0;
}
