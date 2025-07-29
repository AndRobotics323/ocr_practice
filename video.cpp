#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

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
    if (tess.Init(NULL, "kor+eng", tesseract::OEM_LSTM_ONLY)) {
        std::cerr << "Tesseract 초기화 실패\n";
        return 1;
    }
    // tess.SetPageSegMode(tesseract::PSM_SINGLE_WORD);

    cv::VideoCapture cap(dev_num);
    if (!cap.isOpened()) {
        std::cerr << "카메라 열기 실패\n";
        return 1;
    }

    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);



    cv::Mat frame;
    std::cout << "캡처하려면 'c' 누르세요, 종료하려면 ESC(27) 누르세요.\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Live Camera", frame);
        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC 종료

        if (key == 'c' || key == 'C') {
            // 캡처 프레임 복사
            cv::Mat capture = frame.clone();
            cv::Mat gray, processed;


            cv::cvtColor(capture, processed, cv::COLOR_BGR2GRAY);
            // cv::GaussianBlur(processed, processed, cv::Size(3, 3), 0);

            // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
            // clahe->setClipLimit(4.0);
            // clahe->apply(processed, processed);

            cv::threshold(processed, processed, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            // // cv::bitwise_not(processed, processed);

            // cv::resize(processed, processed, cv::Size(), 2.0, 2.0);

            cv::imshow("Processed", processed);
            cv::waitKey(1);

            processed = gray;

            Pix* pix = mat8ToPix(processed);
            tess.SetImage(pix);
            tess.Recognize(0);

            tesseract::ResultIterator* ri = tess.GetIterator();
            tesseract::PageIteratorLevel level = tesseract::RIL_WORD;

            if (ri != nullptr) {
                do {
                    const char* word = ri->GetUTF8Text(level);
                    float conf = ri->Confidence(level);
                    int x1, y1, x2, y2;
                    ri->BoundingBox(level, &x1, &y1, &x2, &y2);

                    int cx = (x1 + x2) / 2;
                    int cy = (y1 + y2) / 2;

                    std::ostringstream oss;
                    oss << word << " (" << cx << ", " << cy << ")";

                    std::string txt = oss.str();

                    if (word ) {
                        std::cout << "detected " << word << "\n" ;

                        cv::rectangle(capture, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                        cv::putText(capture, txt, cv::Point(x1, y1 - 5),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
                    }
                    delete[] word;
                } while (ri->Next(level));
            }

            pixDestroy(&pix);

            // 결과 창 띄우기
            cv::imshow("Captured OCR Result", capture);
            std::cout << "OCR 결과 창을 확인하세요. 아무 키 누르면 다시 라이브로 돌아갑니다.\n";
            cv::waitKey(0);
            cv::destroyWindow("Captured OCR Result");
            std::cout << "라이브 영상으로 복귀합니다.\n";
        }
    }

    tess.End();
    return 0;
}
