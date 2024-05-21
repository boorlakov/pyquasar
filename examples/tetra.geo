h = 1.5;
// Point(1) = {-2, -1, -1, h};
// Point(2) = {1, 0, 0, h};
// Point(3) = {0, 2, 0, h};
// Point(4) = {0, 0, 5, h};

Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {0, 1, 0, h};
Point(4) = {0, 0, 1, h};

Line(1) = {1, 4};
Line(2) = {4, 2};
Line(3) = {2, 1};
Line(4) = {1, 3};
Line(5) = {4, 3};
Line(6) = {2, 3};

Curve Loop(1) = {1, 5, -4};
Plane Surface(1) = {1};

Curve Loop(2) = {2, 6, -5};
Plane Surface(2) = {2};

Curve Loop(3) = {3, 4, -6};
Plane Surface(3) = {3};

Curve Loop(4) = {-3, -1, -2};
Plane Surface(4) = {4};

Surface Loop(1) = {3, 4, 1, 2};
Volume(1) = {1};

Physical Surface("robin", 7) = {2, 4, 3};
Physical Surface("dirichlet", 9) = {1};
Physical Volume("air", 8) = {1};