lc = 0.1;
 
Point(1) = {0, 0, 0, lc};
//+
Point(2) = {1, 0, 0, lc};
//+
Point(3) = {1, 1, 0, lc};
//+
Point(4) = {0, 1, 0, lc};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Point(5) = {0.3, 0.6, 0, lc};
//+
Point(6) = {0.8, 0.2, 0, lc};
//+
Point(7) = {0.8, 0.5, 0, lc};
//+
Point(8) = {0.2, 0.5, 0, lc};
//+
Point(9) = {0.2, 0.2, 0, lc};
//+
Point(10) = {0.5, 0.3, 0, lc};
//+
Point(11) = {0.8, 0.7, 0, lc};
//+
Point(12) = {0.6, 0.3, 0, lc};
//+
Point(13) = {0.6, 0.6, 0, lc};
//+
Point(14) = {0.5, 0.8, 0.0, lc};
//+
Line(5) = {9, 1};
//+
Line(6) = {9, 8};
//+
Line(7) = {8, 4};
//+
Line(8) = {5, 4};
//+
Line(9) = {5, 8};
//+
Line(10) = {5, 14};
//+
Line(11) = {14, 3};
//+
Line(12) = {11, 14};
//+
Line(13) = {14, 4};
//+
Line(14) = {11, 13};
//+
Line(15) = {5, 13};
//+
Line(16) = {13, 10};
//+
Line(17) = {10, 9};
//+
Line(18) = {9, 6};
//+
Line(19) = {6, 2};
//+
Line(20) = {6, 7};
//+
Line(21) = {11, 3};
//+
Line(22) = {7, 12};
//+
Line(23) = {12, 13};
//+
Line(24) = {12, 6};
//+
Line(25) = {12, 10};
//+
Line(26) = {10, 5};
//+
Line(27) = {7, 11};
//+
Curve Loop(1) = {1, -19, -18, 5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {19, 2, -21, -27, -20};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {22, 24, 20};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {23, -14, -27, 22};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {21, -11, -12};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {11, 3, -13};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {13, -8, 10};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {15, -14, 12, -10};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {25, -16, -23};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {15, 16, 26};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {17, 6, -9, -26};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {8, -7, -9};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {7, 4, -5, 6};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {18, -24, 25, 17};
//+
Plane Surface(14) = {14};
//+

Physical Surface(5) = {1};
Physical Surface(6) = {2};
Physical Surface(7) = {3};
Physical Surface(8) = {4};
Physical Surface(9) = {5};
Physical Surface(10) = {6};
Physical Surface(11) = {7};
Physical Surface(12) = {8};
Physical Surface(13) = {9};
Physical Surface(14) = {10};
Physical Surface(15) = {11};
Physical Surface(16) = {12};
Physical Surface(17) = {13};
Physical Surface(18) = {14};

Physical Curve(1) = {1};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {4};
