// This code was created by pygmsh v6.0.2.
p103668 = newp;
Point(p103668) = {-0.5, -0.5, 0.0, 0.06666666666666667};
p103669 = newp;
Point(p103669) = {-0.23838488381881867, -0.5, 0.0, 0.06666666666666667};
p103670 = newp;
Point(p103670) = {-0.5, -0.23838488381881867, 0.0, 0.06666666666666667};
p103671 = newp;
Point(p103671) = {-0.7616151161811813, -0.49999999999999994, 0.0, 0.06666666666666667};
p103672 = newp;
Point(p103672) = {-0.5, -0.7616151161811813, 0.0, 0.06666666666666667};
l84108 = newl;
Ellipse(l84108) = {p103669, p103668, p103669, p103670};
l84109 = newl;
Ellipse(l84109) = {p103670, p103668, p103670, p103671};
l84110 = newl;
Ellipse(l84110) = {p103671, p103668, p103671, p103672};
l84111 = newl;
Ellipse(l84111) = {p103672, p103668, p103672, p103669};
ll21027 = newll;
Line Loop(ll21027) = {l84108, l84109, l84110, l84111};
rs19560 = news;
Surface(rs19560) = {ll21027};
p103673 = newp;
Point(p103673) = {0.5, -0.5, 0.0, 0.06666666666666667};
p103674 = newp;
Point(p103674) = {0.7475799542439852, -0.5, 0.0, 0.06666666666666667};
p103675 = newp;
Point(p103675) = {0.5, -0.25242004575601484, 0.0, 0.06666666666666667};
p103676 = newp;
Point(p103676) = {0.25242004575601484, -0.49999999999999994, 0.0, 0.06666666666666667};
p103677 = newp;
Point(p103677) = {0.49999999999999994, -0.7475799542439852, 0.0, 0.06666666666666667};
l84112 = newl;
Ellipse(l84112) = {p103674, p103673, p103674, p103675};
l84113 = newl;
Ellipse(l84113) = {p103675, p103673, p103675, p103676};
l84114 = newl;
Ellipse(l84114) = {p103676, p103673, p103676, p103677};
l84115 = newl;
Ellipse(l84115) = {p103677, p103673, p103677, p103674};
ll21028 = newll;
Line Loop(ll21028) = {l84112, l84113, l84114, l84115};
rs19561 = news;
Surface(rs19561) = {ll21028};
p103678 = newp;
Point(p103678) = {-0.5, 0.5, 0.0, 0.06666666666666667};
p103679 = newp;
Point(p103679) = {-0.16651854838418195, 0.5, 0.0, 0.06666666666666667};
p103680 = newp;
Point(p103680) = {-0.5, 0.833481451615818, 0.0, 0.06666666666666667};
p103681 = newp;
Point(p103681) = {-0.833481451615818, 0.5, 0.0, 0.06666666666666667};
p103682 = newp;
Point(p103682) = {-0.5000000000000001, 0.16651854838418195, 0.0, 0.06666666666666667};
l84116 = newl;
Ellipse(l84116) = {p103679, p103678, p103679, p103680};
l84117 = newl;
Ellipse(l84117) = {p103680, p103678, p103680, p103681};
l84118 = newl;
Ellipse(l84118) = {p103681, p103678, p103681, p103682};
l84119 = newl;
Ellipse(l84119) = {p103682, p103678, p103682, p103679};
ll21029 = newll;
Line Loop(ll21029) = {l84116, l84117, l84118, l84119};
rs19562 = news;
Surface(rs19562) = {ll21029};
p103683 = newp;
Point(p103683) = {0.5, 0.5, 0.0, 0.06666666666666667};
p103684 = newp;
Point(p103684) = {0.8450382402970926, 0.5, 0.0, 0.06666666666666667};
p103685 = newp;
Point(p103685) = {0.5, 0.8450382402970926, 0.0, 0.06666666666666667};
p103686 = newp;
Point(p103686) = {0.15496175970290743, 0.5, 0.0, 0.06666666666666667};
p103687 = newp;
Point(p103687) = {0.49999999999999994, 0.15496175970290743, 0.0, 0.06666666666666667};
l84120 = newl;
Ellipse(l84120) = {p103684, p103683, p103684, p103685};
l84121 = newl;
Ellipse(l84121) = {p103685, p103683, p103685, p103686};
l84122 = newl;
Ellipse(l84122) = {p103686, p103683, p103686, p103687};
l84123 = newl;
Ellipse(l84123) = {p103687, p103683, p103687, p103684};
ll21030 = newll;
Line Loop(ll21030) = {l84120, l84121, l84122, l84123};
rs19563 = news;
Surface(rs19563) = {ll21030};
p103688 = newp;
Point(p103688) = {-1.0, -1.0, 0.0, 0.06666666666666667};
p103689 = newp;
Point(p103689) = {1.0, -1.0, 0.0, 0.06666666666666667};
p103690 = newp;
Point(p103690) = {1.0, 1.0, 0.0, 0.06666666666666667};
p103691 = newp;
Point(p103691) = {-1.0, 1.0, 0.0, 0.06666666666666667};
l84124 = newl;
Line(l84124) = {p103688, p103689};
l84125 = newl;
Line(l84125) = {p103689, p103690};
l84126 = newl;
Line(l84126) = {p103690, p103691};
l84127 = newl;
Line(l84127) = {p103691, p103688};
ll21031 = newll;
Line Loop(ll21031) = {l84124, l84125, l84126, l84127};
s1467 = news;
Plane Surface(s1467) = {ll21031,ll21027,ll21028,ll21029,ll21030};
Physical Surface(1) = {s1467};
Physical Surface(0) = {rs19560, rs19561, rs19562, rs19563};
Physical Line(2) = {l84124, l84125, l84126, l84127};
Transfinite Line {l84124, l84125, l84126, l84127} = 31;