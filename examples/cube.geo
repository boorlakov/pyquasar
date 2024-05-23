SetFactory("OpenCASCADE");

Point(1) = {0, 0, 0, 1.0};
Point(2) = {4, 0, 0, 1.0};
Point(3) = {4, 6, 0, 1.0};
Point(4) = {0, 6, 0, 1.0};
Point(5) = {0, 0, 3, 1.0};
Point(6) = {4, 0, 3, 1.0};
Point(7) = {4, 6, 3, 1.0};
Point(8) = {0, 6, 3, 1.0};
Point(9) = {1, 1, 1, 0.1};
Point(10) = {1.5, 1, 1, 0.1};
Point(11) = {1.5, 1, 1.5, 0.1};
Point(12) = {1, 1, 1.5, 0.1};

Line(1) = {4, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
Line(5) = {4, 8};
Line(6) = {8, 7};
Line(7) = {7, 3};
Line(8) = {7, 6};
Line(9) = {6, 2};
Line(10) = {1, 5};
Line(11) = {5, 6};
Line(12) = {5, 8};
Line(13) = {9, 10};
Line(14) = {10, 11};
Line(15) = {11, 12};
Line(16) = {12, 9};

Curve Loop(1) = {1, 10, 12, -5};
Plane Surface(1) = {1};

Curve Loop(2) = {-10, -11, -9, 2};
Plane Surface(2) = {2};

Curve Loop(3) = {9, 3, -7, 8};
Plane Surface(3) = {3};

Curve Loop(4) = {4, 5, 6, 7};
Plane Surface(4) = {4};

Curve Loop(5) = {-6, -8, 11, -12};
Plane Surface(5) = {5};

Curve Loop(6) = {-2, -3, -4, -1};
Plane Surface(6) = {6};

Curve Loop(7) = {13, 14, 15, 16};
Plane Surface(7) = {7};

Surface Loop(1) = {1, 6, 2, 3, 4, 5};
Volume(1) = {1};

Physical Volume("air", 13) = {1};

Physical Surface("source", 17) = {7};

BooleanFragments{ Volume{1}; Delete; }{ Surface{7}; Delete; }

//Physical Surface("dirichlet", 14) = {8};
Physical Surface("neumann", 29) = {8, 12, 11, 13, 9, 10};
