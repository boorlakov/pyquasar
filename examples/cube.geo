// Gmsh project created on Wed Mar  6 21:16:02 2024
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {0, 1, 0, 1.0};
//+
Point(4) = {0, 0, 1, 1.0};
//+
Point(5) = {0, 1, 1, 1.0};
//+
Point(6) = {1, 1, 1, 1.0};
//+
Point(7) = {1, 0, 1, 1.0};
//+
Point(8) = {1, 1, 0, 1.0};//+
Line(1) = {1, 4};
//+
Line(2) = {4, 7};
//+
Line(3) = {7, 8};
//+
Line(4) = {8, 3};
//+
Line(5) = {3, 1};
//+
Line(6) = {1, 2};
//+
Line(7) = {2, 7};
//+
Line(8) = {2, 8};
//+
Line(9) = {3, 4};
//+
Line(10) = {4, 5};
//+
Line(11) = {5, 3};
//+
Line(12) = {5, 6};
//+
Line(13) = {6, 7};
//+
Line(14) = {6, 8};
//+
Curve Loop(1) = {-1, -2, 7, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {-7, -3, 8};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {-6, -8, -4, -5};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {9, 2, 3, 4};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {2, -13, -12, -10};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {-11, 4, 14, 12};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {3, -14, 13};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {1, -9, 5};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {9, 10, 11};
//+
Plane Surface(9) = {9};
//+
Surface Loop(1) = {8, 1, 2, 3, 4};
//+
Volume(1) = {1};
//+
Surface Loop(2) = {4, 9, 5, 7, 6};
//+
Volume(2) = {2};
//+
Physical Surface("dirichlet", 15) = {1, 3, 5, 6};
//+
Physical Surface("neumann", 16) = {8, 9, 7, 2};
//+
Physical Volume("steel", 17) = {1};
//+
Physical Volume("air", 18) = {2};
