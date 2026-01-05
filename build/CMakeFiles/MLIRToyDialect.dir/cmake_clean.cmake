file(REMOVE_RECURSE
  "ToyOps.cpp.inc"
  "ToyOps.h.inc"
  "ToyOpsDialect.cpp.inc"
  "ToyOpsDialect.h.inc"
  "libMLIRToyDialect.a"
  "libMLIRToyDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRToyDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
