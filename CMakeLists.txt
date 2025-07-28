cmake_minimum_required(VERSION 3.10)
project(ocr_project)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(TESSERACT REQUIRED tesseract)
pkg_check_modules(LEPT REQUIRED lept)

include_directories(${OpenCV_INCLUDE_DIRS} ${TESSERACT_INCLUDE_DIRS})

add_executable(ocr_image image.cpp)
add_executable(ocr_video video.cpp)

target_link_libraries(ocr_video 
    ${OpenCV_LIBS} 
    ${TESSERACT_LIBRARIES}
    ${LEPT_LIBRARIES}    
)


target_link_libraries(ocr_image
    ${OpenCV_LIBS} 
    ${TESSERACT_LIBRARIES}
    ${LEPT_LIBRARIES}    
)
