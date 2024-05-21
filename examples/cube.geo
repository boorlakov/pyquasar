//+
SetFactory("Built-in");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {4, 0, 0, 1.0};
//+
Point(3) = {4, 6, 0, 1.0};
//+
Point(4) = {0, 6, 0, 1.0};
//+
Point(5) = {0, 0, 3, 1.0};
//+
Point(6) = {4, 0, 3, 1.0};
//+
Point(7) = {4, 6, 3, 1.0};
//+
Point(8) = {0, 6, 3, 1.0};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 8};
//+
Line(6) = {8, 7};
//+
Line(7) = {7, 3};
//+
Line(8) = {7, 6};
//+
Line(9) = {6, 2};
//+
Line(10) = {1, 5};
//+
Line(11) = {5, 6};
//+
Line(12) = {5, 8};
//+
Curve Loop(1) = {1, 10, 12, -5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {-10, -11, -9, 2};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {9, 3, -7, 8};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {4, 5, 6, 7};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {-6, -8, 11, -12};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {-2, -3, -4, -1};
//+
Plane Surface(6) = {6};
//+
Surface Loop(1) = {1, 6, 2, 3, 4, 5};
//+
Volume(1) = {1};
//+
Physical Volume("air", 13) = {1};
//+
Physical Surface("dirichlet", 14) = {2, 1, 4, 3};
//+
Physical Surface("robin", 15) = {5, 6};