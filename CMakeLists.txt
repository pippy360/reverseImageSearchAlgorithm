cmake_minimum_required(VERSION 3.0)
project(imageHashTesting)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(Boost REQUIRED)

#find_package(hiredis REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)

find_package(OpenMP REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

################################
# Testing
################################
add_subdirectory(googletest)

enable_testing()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

include_directories(src)

include_directories(include)

SET(TARGET_H
        src/img_hash/AverageHash.h
        src/img_hash/BlockMeanHash.h
        src/img_hash/PerceptualHash.h
        src/img_hash/FragmentHash.h
        src/Keypoint.h
        src/ShapeAndPositionInvariantImage.h
        src/Triangle.h
	src/mainImageProcessingFunctions.hpp        
        )

set(IMAGE_HASH_FILES 
        src/img_hash/img_hash_opencv_module/average_hash.cpp
        src/img_hash/img_hash_opencv_module/block_mean_hash.cpp
        src/img_hash/img_hash_opencv_module/color_moment_hash.cpp
        src/img_hash/img_hash_opencv_module/marr_hildreth_hash.cpp
        src/img_hash/img_hash_opencv_module/phash.cpp
        src/img_hash/img_hash_opencv_module/PHash_Fast.cpp
        )


set(SOURCE_FILE 
        src/main.cc
	${IMAGE_HASH_FILES} src/img_hash/img_hash_opencv_module/PHash_Fast.cpp src/img_hash/img_hash_opencv_module/PHash_Fast.h)

set(TEST_FILES
        test/averageHashTest.cpp 
        test/blockMeanHashTest.cpp 
        test/perceptualHashTest.cpp 
	${IMAGE_HASH_FILES}
		test/fullKeypointAccuracyTest.cpp
		)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

#set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

add_executable(app ${SOURCE_FILE} ${TARGET_H})
add_executable(averageHashTest ${TEST_FILES} ${TARGET_H})
add_executable(utilsTest test/utilsTest.cpp  ${TARGET_H})
target_link_libraries(utilsTest gtest gtest_main ${OpenCV_LIBS})

target_link_libraries(averageHashTest gtest gtest_main ${OpenCV_LIBS})
link_directories(lib)
target_link_libraries(app ${OpenCV_LIBS})
target_link_libraries(app ${CMAKE_CURRENT_SOURCE_DIR}/lib/libhiredis.a)

# Options. Turn on with 'cmake -Dmyvarname=ON'.


set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -g -Wall")



get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
