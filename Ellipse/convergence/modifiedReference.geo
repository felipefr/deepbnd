lc = 0.125; 
Lx1 = 0.25; 
Lx2 = 0.5; 
Lx3 = 0.25; 
Ly1 = 0.25; 
Ly2 = 0.5; 
Ly3 = 0.25; 
theta = 0.5235987755982988; 
alpha = 0.0; 
LxE = 0.2; 
LyE = 0.1; 
Nx1 = 3; 
Nx2 = 5; 
Nx3 = 3; 
Ny1 = 3; 
Ny2 = 5; 
Ny3 = 3; 
halfPi = 0.5*Pi;
Lx = Lx1 + Lx2 + Lx3;
Ly = Ly1 + Ly2 + Ly3;
x0 = Lx1 + 0.5*Lx2;
y0 = Ly1 + 0.5*Ly2;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {Lx, 0.0, 0.0, lc};
Point(3) = {Lx, Ly, 0.0, lc};
Point(4) = {0.0, Ly, 0.0, lc};
Point(5) = {Lx1, 0.0, 0.0, lc};
Point(6) = {Lx1 + Lx2, 0.0, 0.0, lc};
Point(7) = {Lx, Ly1, 0.0, lc};
Point(8) = {Lx, Ly1 + Ly2, 0.0, lc};
Point(9) = {Lx1 + Lx2, Ly, 0.0, lc};
Point(10) = {Lx1, Ly, 0.0, lc};
Point(11) = {0.0, Ly1 + Ly2, 0.0, lc};
Point(12) = {0.0, Ly1, 0.0, lc};
Point(13) = {Lx1, Ly1, 0.0, lc};
Point(14) = {Lx1 + Lx2, Ly1, 0.0, lc};
Point(15) = {Lx1 + Lx2, Ly1 + Ly2, 0.0, lc};
Point(16) = {Lx1, Ly1 + Ly2, 0.0, lc};

Point(17) = {x0, y0, 0.0, lc};
Point(18) = {x0 + LxE, y0, 0.0, lc};
Point(19) = {x0, y0 + LyE, 0.0, lc};
Point(20) = {x0 - LxE, y0, 0.0, lc};
Point(21) = {x0, y0-LyE, 0.0, lc};

Line(1) = {1,5};
Line(2) = {5,6};
Line(3) = {6,2};
Line(4) = {2,7};
Line(5) = {7,8};
Line(6) = {8,3};
Line(7) = {3,9};
Line(8) = {9,10};
Line(9) = {10,4};
Line(10) = {4,11};
Line(11) = {11,12};
Line(12) = {12,1};
Line(13) = {13,14};
Line(14) = {14,15};
Line(15) = {15,16};
Line(16) = {16,13};
Line(17) = {5,13};
Line(18) = {6,14};
Line(19) = {10,16};
Line(20) = {9,15};
Line(21) = {7,14};
Line(22) = {8,15};
Line(23) = {12,13};
Line(24) = {11,16};

Ellipse(25) = {18,17,19};
Ellipse(26) = {19,17,20};
Ellipse(27) = {20,17,21};
Ellipse(28) = {21,17,18};

Curve Loop(1) = {1, 17, -23, 12};
Plane Surface(1) = {1};
Curve Loop(2) = {2, 18, -13, -17};
Plane Surface(2) = {2};
Curve Loop(3) = {3, 4, 21, -18};
Plane Surface(3) = {3};
Curve Loop(4) = {23, -16, -24, 11};
Plane Surface(4) = {4};
Curve Loop(5) = {5, 22, -14, -21};
Plane Surface(5) = {5};
Curve Loop(6) = {24, -19, 9, 10};
Plane Surface(6) = {6};
Curve Loop(7) = {15, -19, -8, 20};
Plane Surface(7) = {7};
Curve Loop(8) = {22, -20, -7, -6};
Plane Surface(8) = {8};
Curve Loop(9) = {16, 13, 14, 15};
Curve Loop(10) = {25, 26, 27, 28};
Plane Surface(9) = {9, 10};
Plane Surface(10) = {10};

Rotate {{0, 0, 1}, {x0, y0, 0}, theta} { Surface{10}; }
Rotate {{0, 0, 1}, {x0, y0, 0}, alpha} {Surface{10}; Surface{9}; }


Physical Curve("Left") = {10, 11, 12};
Physical Curve("Bottom") = {1, 2, 3};
Physical Curve("Right_1") = {4};
Physical Curve("Right_2") = {5};
Physical Curve("Right_3") = {6};


//Physical Curve("Top") = {7, 8, 9};
//Physical Curve("Ellipse") = {27, 28, 25, 26};

Physical Surface("left_bottom") = {1};
Physical Surface("middle_bottom") = {2};
Physical Surface("right_bottom") = {3};
Physical Surface("left_middle") = {4};
Physical Surface("right_middle") = {5};
Physical Surface("left_top") = {6};
Physical Surface("middle_top") = {7};
Physical Surface("right_top") = {8};
Physical Surface("middle_out_ellipse") = {9};
Physical Surface("ellipse_vol") = {10};

Transfinite Curve {1,23,24,9} = Nx1 Using Progression 1;
Transfinite Curve {2,13,15,8} = Nx2 Using Progression 1;
Transfinite Curve {3,21,22,7} = Nx3 Using Progression 1;
Transfinite Curve {12,17,18,4} = Ny1 Using Progression 1;
Transfinite Curve {11,16,14,5} = Ny2 Using Progression 1;
Transfinite Curve {10,19,20,6} = Ny3 Using Progression 1;
Transfinite Surface {1,2,3,4,5,6,7,8} Alternated;

