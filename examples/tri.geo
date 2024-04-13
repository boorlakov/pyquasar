Point(newp) = {0, 0, 0, 1.5};
Point(newp) = {1, 0, 0, 1.5};
Point(newp) = {0, 1, 0, 1.5};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};

Curve Loop(1) = {3, 1, 2};
Plane Surface(1) = {1};

Physical Surface("steel", 6) = {1};
Physical Curve("dirichlet", 7) = {1, 2, 3};