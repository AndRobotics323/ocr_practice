#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>




Pix* mat8ToPix(const cv::Mat& mat) {
    int width = mat.cols;
    int height = mat.rows;
    Pix* pix = pixCreate(width, height, 8);
    l_uint32* data = pixGetData(pix);
    int wpl = pixGetWpl(pix);
    for (int y = 0; y < height; ++y) {
        l_uint32* line = data + y * wpl;
        const uchar* row = mat.ptr(y);
        for (int x = 0; x < width; ++x) {
            SET_DATA_BYTE(line, x, row[x]);
        }
    }
    return pix;
}

int main() {

    int dev_num = 0;

    tesseract::TessBaseAPI tess;
    if (tess.Init(NULL, "kor", tesseract::OEM_LSTM_ONLY)) {
        std::cerr << "Tesseract 초기화 실패\n";
        return 1;
    }
    tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);  // 전체 블록 자동 감지

    cv::VideoCapture cap(dev_num);  
    if (!cap.isOpened()) {
        std::cerr << "카메라 열기 실패\n";
        return 1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // cv::Mat gray;
        // cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);


        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        // 3. 이진화 (선택적 전처리, 명도 임계값 자동 계산)
        cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);



        Pix* pix = mat8ToPix(gray);
        tess.SetImage(pix);
        tess.Recognize(0);

        tesseract::ResultIterator* ri = tess.GetIterator();
        tesseract::PageIteratorLevel level = tesseract::RIL_WORD;

        if (ri != nullptr) {
            do {
                // const char* word = ri->GetUTF8Text(level);
                const char* word = ri->GetUTF8Text(level);

                float conf = ri->Confidence(level);
                int x1, y1, x2, y2;
                int cx = (x1 + x2) / 2;
                int cy = (y1 + y2) / 2;

                ri->BoundingBox(level, &x1, &y1, &x2, &y2);
                
                // std::ostringstream txt;
                // std::string txt =  std::to_string(cx) + " , " + std::to_string(cy) + ")";

                std::ostringstream oss;
                oss << word << " at (" << cx << " , " << cy << ")";
                std::string txt = oss.str();
                
                if (word ) {
                    // Draw box and text
                    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,255,0), 2);
                    cv::putText(frame, txt, cv::Point(x1, y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);
                    
                    
                    // cv::putText(frame, word, cv::Point(x1, y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);

                    delete[] word;
                }
            } while (ri->Next(level));
        }

        cv::imshow("OCR Result", frame);
        if (cv::waitKey(1) == 27 || cv::waitKey(1) == 'q') break;  // 'q'나 ESC to quit

        pixDestroy(&pix);
    }

    tess.End();
    return 0;
}
