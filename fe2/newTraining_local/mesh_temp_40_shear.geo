// This code was created by pygmsh v6.0.2.
p2120 = newp;
Point(p2120) = {-0.5, -0.5, 0.0, 0.06666666666666667};
p2121 = newp;
Point(p2121) = {-0.20530176927374072, -0.5, 0.0, 0.06666666666666667};
p2122 = newp;
Point(p2122) = {-0.5, -0.20530176927374072, 0.0, 0.06666666666666667};
p2123 = newp;
Point(p2123) = {-0.7946982307262593, -0.49999999999999994, 0.0, 0.06666666666666667};
p2124 = newp;
Point(p2124) = {-0.5, -0.7946982307262593, 0.0, 0.06666666666666667};
l1720 = newl;
Ellipse(l1720) = {p2121, p2120, p2121, p2122};
l1721 = newl;
Ellipse(l1721) = {p2122, p2120, p2122, p2123};
l1722 = newl;
Ellipse(l1722) = {p2123, p2120, p2123, p2124};
l1723 = newl;
Ellipse(l1723) = {p2124, p2120, p2124, p2121};
ll430 = newll;
Line Loop(ll430) = {l1720, l1721, l1722, l1723};
rs400 = news;
Surface(rs400) = {ll430};
p2125 = newp;
Point(p2125) = {0.5, -0.5, 0.0, 0.06666666666666667};
p2126 = newp;
Point(p2126) = {0.7796094563105999, -0.5, 0.0, 0.06666666666666667};
p2127 = newp;
Point(p2127) = {0.5, -0.22039054368940014, 0.0, 0.06666666666666667};
p2128 = newp;
Point(p2128) = {0.22039054368940014, -0.49999999999999994, 0.0, 0.06666666666666667};
p2129 = newp;
Point(p2129) = {0.49999999999999994, -0.7796094563105999, 0.0, 0.06666666666666667};
l1724 = newl;
Ellipse(l1724) = {p2126, p2125, p2126, p2127};
l1725 = newl;
Ellipse(l1725) = {p2127, p2125, p2127, p2128};
l1726 = newl;
Ellipse(l1726) = {p2128, p2125, p2128, p2129};
l1727 = newl;
Ellipse(l1727) = {p2129, p2125, p2129, p2126};
ll431 = newll;
Line Loop(ll431) = {l1724, l1725, l1726, l1727};
rs401 = news;
Surface(rs401) = {ll431};
p2130 = newp;
Point(p2130) = {-0.5, 0.5, 0.0, 0.06666666666666667};
p2131 = newp;
Point(p2131) = {-0.16110979907362055, 0.5, 0.0, 0.06666666666666667};
p2132 = newp;
Point(p2132) = {-0.5, 0.8388902009263794, 0.0, 0.06666666666666667};
p2133 = newp;
Point(p2133) = {-0.8388902009263794, 0.5, 0.0, 0.06666666666666667};
p2134 = newp;
Point(p2134) = {-0.5000000000000001, 0.16110979907362055, 0.0, 0.06666666666666667};
l1728 = newl;
Ellipse(l1728) = {p2131, p2130, p2131, p2132};
l1729 = newl;
Ellipse(l1729) = {p2132, p2130, p2132, p2133};
l1730 = newl;
Ellipse(l1730) = {p2133, p2130, p2133, p2134};
l1731 = newl;
Ellipse(l1731) = {p2134, p2130, p2134, p2131};
ll432 = newll;
Line Loop(ll432) = {l1728, l1729, l1730, l1731};
rs402 = news;
Surface(rs402) = {ll432};
p2135 = newp;
Point(p2135) = {0.5, 0.5, 0.0, 0.06666666666666667};
p2136 = newp;
Point(p2136) = {0.7830627228400721, 0.5, 0.0, 0.06666666666666667};
p2137 = newp;
Point(p2137) = {0.5, 0.7830627228400721, 0.0, 0.06666666666666667};
p2138 = newp;
Point(p2138) = {0.21693727715992794, 0.5, 0.0, 0.06666666666666667};
p2139 = newp;
Point(p2139) = {0.49999999999999994, 0.21693727715992794, 0.0, 0.06666666666666667};
l1732 = newl;
Ellipse(l1732) = {p2136, p2135, p2136, p2137};
l1733 = newl;
Ellipse(l1733) = {p2137, p2135, p2137, p2138};
l1734 = newl;
Ellipse(l1734) = {p2138, p2135, p2138, p2139};
l1735 = newl;
Ellipse(l1735) = {p2139, p2135, p2139, p2136};
ll433 = newll;
Line Loop(ll433) = {l1732, l1733, l1734, l1735};
rs403 = news;
Surface(rs403) = {ll433};
p2140 = newp;
Point(p2140) = {-1.0, -1.0, 0.0, 0.06666666666666667};
p2141 = newp;
Point(p2141) = {1.0, -1.0, 0.0, 0.06666666666666667};
p2142 = newp;
Point(p2142) = {1.0, 1.0, 0.0, 0.06666666666666667};
p2143 = newp;
Point(p2143) = {-1.0, 1.0, 0.0, 0.06666666666666667};
l1736 = newl;
Line(l1736) = {p2140, p2141};
l1737 = newl;
Line(l1737) = {p2141, p2142};
l1738 = newl;
Line(l1738) = {p2142, p2143};
l1739 = newl;
Line(l1739) = {p2143, p2140};
ll434 = newll;
Line Loop(ll434) = {l1736, l1737, l1738, l1739};
s30 = news;
Plane Surface(s30) = {ll434,ll430,ll431,ll432,ll433};
Physical Surface(1) = {s30};
Physical Surface(0) = {rs400, rs401, rs402, rs403};
Physical Line(2) = {l1736, l1737, l1738, l1739};
Transfinite Line {l1736, l1737, l1738, l1739} = 31;