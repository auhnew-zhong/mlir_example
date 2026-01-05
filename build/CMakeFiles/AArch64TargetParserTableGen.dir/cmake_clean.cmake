file(REMOVE_RECURSE
  "ToyOps.cpp.inc"
  "ToyOps.h.inc"
  "ToyOpsDialect.cpp.inc"
  "ToyOpsDialect.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/AArch64TargetParserTableGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
