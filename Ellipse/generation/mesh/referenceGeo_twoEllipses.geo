Lx = Lx1 + Lx2 + Lx3;
Ly = Ly1 + Ly2 + Ly3;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {Lx, 0.0, 0.0, lc};
Point(3) = {Lx, Ly, 0.0, lc};
Point(4) = {0.0, Ly, 0.0, lc};

Point(5) = {x0, y0, 0.0, lc};
Point(6) = {x0 + LxE, y0, 0.0, lc};
Point(7) = {x0, y0 + LyE, 0.0, lc};
Point(8) = {x0 - LxE, y0, 0.0, lc};
Point(9) = {x0, y0-LyE, 0.0, lc};

Point(10) = {x1, y1, 0.0, lc};
Point(11) = {x1 + LxE, y1, 0.0, lc};
Point(12) = {x1, y1 + LyE, 0.0, lc};
Point(13) = {x1 - LxE, y1, 0.0, lc};
Point(14) = {x1, y1-LyE, 0.0, lc};

Ellipse(15) = {6,5,7};
Ellipse(16) = {7,5,8};
Ellipse(17) = {8,5,9};
Ellipse(18) = {9,5,6};

Ellipse(19) = {11,10,12};
Ellipse(20) = {12,10,13};
Ellipse(21) = {13,10,14};
Ellipse(22) = {14,10,11};

Line(23) = {1, 2};
Line(24) = {2, 3};
Line(25) = {3, 4};
Line(26) = {4, 1};
Curve Loop(1) = {25, 26, 23, 24};
Curve Loop(2) = {15, 16, 17, 18};
Curve Loop(3) = {19, 20, 21, 22};
Plane Surface(1) = {1, 2, 3};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

Rotate {{0, 0, 1}, {x0, y0, 0}, theta} { Surface{2}; }
Rotate {{0, 0, 1}, {x1, y1, 0}, alpha} {Surface{3}; }

Physical Curve("Left") = {26};
Physical Curve("Bottom") = {23};
Physical Curve("Right") = {24};
Physical Curve("Top") = {25};

Physical Surface("base_vol") = {1};
Physical Surface("ellipse_vol1") = {2};
Physical Surface("ellipse_vol2") = {3};