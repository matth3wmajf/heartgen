cmake_minimum_required(VERSION 3.10)
project(HeartGenerator)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.c)

include_directories(include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/alpha)

add_executable(heartgen ${SOURCES})

target_link_libraries(heartgen PUBLIC alpha)

if(NOT WIN32)
	target_link_libraries(heartgen PUBLIC m)
endif()