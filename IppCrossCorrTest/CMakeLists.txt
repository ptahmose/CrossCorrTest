# CMakeList.txt : CMake project for CMakeProject1, include source and define
# project specific logic here.
#

find_package(IPP REQUIRED)

set(OpenCV_DIR "D:/dev/Github/vcpkg/installed/x64-windows/share/opencv4")
find_package(OpenCV 4.2 REQUIRED)


message("IPP -> ${IPP_INCLUDE_DIRS}")
# Add source to this project's executable.
add_executable (IppCrossCorrTest "IppCrossCorrTest.cpp" "IppCrossCorrTest.h")

message("OPENCV -> ${OpenCV_INCLUDE_DIRS}")
target_include_directories(IppCrossCorrTest PRIVATE ${IPP_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

IF(WIN32) 
    set(IPP_LIBRARYNAME_SUFFIX "mt")
ENDIF()

target_link_libraries(IppCrossCorrTest PRIVATE
        "${IPP_LIB_SEARCHPATH}/${CMAKE_STATIC_LIBRARY_PREFIX}ippi${IPP_LIBRARYNAME_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
        "${IPP_LIB_SEARCHPATH}/${CMAKE_STATIC_LIBRARY_PREFIX}ippcc${IPP_LIBRARYNAME_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
        "${IPP_LIB_SEARCHPATH}/${CMAKE_STATIC_LIBRARY_PREFIX}ippcore${IPP_LIBRARYNAME_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
        "${IPP_LIB_SEARCHPATH}/${CMAKE_STATIC_LIBRARY_PREFIX}ippvm${IPP_LIBRARYNAME_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
        "${IPP_LIB_SEARCHPATH}/${CMAKE_STATIC_LIBRARY_PREFIX}ipps${IPP_LIBRARYNAME_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
          ${OpenCV_LIBS}
        )


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET IppCrossCorrTest  PROPERTY CXX_STANDARD 17)
endif()

# TODO: Add tests and install targets if needed.
