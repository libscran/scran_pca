include(FetchContent)

FetchContent_Declare(
  tatami
  GIT_REPOSITORY https://github.com/tatami-inc/tatami
  GIT_TAG master # ^4.0.0
)

FetchContent_Declare(
  tatami_stats
  GIT_REPOSITORY https://github.com/tatami-inc/tatami_stats
  GIT_TAG master # ^2.0.0
)

FetchContent_Declare(
  irlba
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG master # ^2.0.0
)

FetchContent_Declare(
  eigen 
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG 3.4.0
)

FetchContent_Declare(
  scran_blocks
  GIT_REPOSITORY https://github.com/libscran/scran_blocks
  GIT_TAG master # ^0.1.0
)

FetchContent_Declare(
  sanisizer 
  GIT_REPOSITORY https://github.com/LTLA/sanisizer
  GIT_TAG master # ^0.1.2
)

FetchContent_MakeAvailable(tatami)
FetchContent_MakeAvailable(tatami_stats)
FetchContent_MakeAvailable(irlba)
FetchContent_MakeAvailable(eigen)
FetchContent_MakeAvailable(scran_blocks)
FetchContent_MakeAvailable(sanisizer)
