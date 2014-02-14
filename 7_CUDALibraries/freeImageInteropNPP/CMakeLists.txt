project(FreeImageInteropNPP)

include_directories(../../common/FreeImage/include ../../common/UtilNPP/)


SET(FreeImageInteropNPP_SOURCES
freeImageInteropNPP.cpp
)


SET(LIBRARY_OUTPUT_PATH ../../bin/)
SET(EXECUTABLE_OUTPUT_PATH ../../bin/)


CUDA_ADD_EXECUTABLE(FreeImageInterop 
${FreeImageInterop_SOURCES}
)

TARGET_LINK_LIBRARIES(FreeImageInteropNPP
npp
UtilNPP
FreeImage
)


add_library(FreeImage UNKNOWN IMPORTED)
set_property(TARGET FreeImage PROPERTY IMPORTED_LOCATION ../../common/FreeImage/lib/FreeImage.lib)

IF(CMAKE_CXX_COMPILER MATCHES cl)

SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /0d")
SET(CMAKE_CXX_FLAGS_RELEASE "/MT /O2")

ENDIF(CMAKE_CXX_COMPILER MATCHES cl)