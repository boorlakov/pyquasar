//+
SetFactory("OpenCASCADE");
Cylinder(1) = {0, 0, 0, 0, 0, 3, 1, 2*Pi};
//+
Physical Surface("neumann", 4) = {1, 3, 2};
//+
Physical Volume("Air", 5) = {1};
//+