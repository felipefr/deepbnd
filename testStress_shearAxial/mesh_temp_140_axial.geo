// This code was created by pygmsh v6.0.2.
p7576 = newp;
Point(p7576) = {-0.5, -0.5, 0.0, 0.06666666666666667};
p7577 = newp;
Point(p7577) = {-0.19888718577876902, -0.5, 0.0, 0.06666666666666667};
p7578 = newp;
Point(p7578) = {-0.5, -0.19888718577876902, 0.0, 0.06666666666666667};
p7579 = newp;
Point(p7579) = {-0.8011128142212309, -0.49999999999999994, 0.0, 0.06666666666666667};
p7580 = newp;
Point(p7580) = {-0.5, -0.8011128142212309, 0.0, 0.06666666666666667};
l6164 = newl;
Ellipse(l6164) = {p7577, p7576, p7577, p7578};
l6165 = newl;
Ellipse(l6165) = {p7578, p7576, p7578, p7579};
l6166 = newl;
Ellipse(l6166) = {p7579, p7576, p7579, p7580};
l6167 = newl;
Ellipse(l6167) = {p7580, p7576, p7580, p7577};
ll1541 = newll;
Line Loop(ll1541) = {l6164, l6165, l6166, l6167};
rs1412 = news;
Surface(rs1412) = {ll1541};
p7581 = newp;
Point(p7581) = {0.5, -0.5, 0.0, 0.06666666666666667};
p7582 = newp;
Point(p7582) = {0.8555346911312431, -0.5, 0.0, 0.06666666666666667};
p7583 = newp;
Point(p7583) = {0.5, -0.14446530886875686, 0.0, 0.06666666666666667};
p7584 = newp;
Point(p7584) = {0.14446530886875686, -0.49999999999999994, 0.0, 0.06666666666666667};
p7585 = newp;
Point(p7585) = {0.49999999999999994, -0.8555346911312431, 0.0, 0.06666666666666667};
l6168 = newl;
Ellipse(l6168) = {p7582, p7581, p7582, p7583};
l6169 = newl;
Ellipse(l6169) = {p7583, p7581, p7583, p7584};
l6170 = newl;
Ellipse(l6170) = {p7584, p7581, p7584, p7585};
l6171 = newl;
Ellipse(l6171) = {p7585, p7581, p7585, p7582};
ll1542 = newll;
Line Loop(ll1542) = {l6168, l6169, l6170, l6171};
rs1413 = news;
Surface(rs1413) = {ll1542};
p7586 = newp;
Point(p7586) = {-0.5, 0.5, 0.0, 0.06666666666666667};
p7587 = newp;
Point(p7587) = {-0.25762355490497835, 0.5, 0.0, 0.06666666666666667};
p7588 = newp;
Point(p7588) = {-0.5, 0.7423764450950217, 0.0, 0.06666666666666667};
p7589 = newp;
Point(p7589) = {-0.7423764450950217, 0.5, 0.0, 0.06666666666666667};
p7590 = newp;
Point(p7590) = {-0.5, 0.25762355490497835, 0.0, 0.06666666666666667};
l6172 = newl;
Ellipse(l6172) = {p7587, p7586, p7587, p7588};
l6173 = newl;
Ellipse(l6173) = {p7588, p7586, p7588, p7589};
l6174 = newl;
Ellipse(l6174) = {p7589, p7586, p7589, p7590};
l6175 = newl;
Ellipse(l6175) = {p7590, p7586, p7590, p7587};
ll1543 = newll;
Line Loop(ll1543) = {l6172, l6173, l6174, l6175};
rs1414 = news;
Surface(rs1414) = {ll1543};
p7591 = newp;
Point(p7591) = {0.5, 0.5, 0.0, 0.06666666666666667};
p7592 = newp;
Point(p7592) = {0.7901368366377082, 0.5, 0.0, 0.06666666666666667};
p7593 = newp;
Point(p7593) = {0.5, 0.7901368366377082, 0.0, 0.06666666666666667};
p7594 = newp;
Point(p7594) = {0.20986316336229188, 0.5, 0.0, 0.06666666666666667};
p7595 = newp;
Point(p7595) = {0.49999999999999994, 0.20986316336229188, 0.0, 0.06666666666666667};
l6176 = newl;
Ellipse(l6176) = {p7592, p7591, p7592, p7593};
l6177 = newl;
Ellipse(l6177) = {p7593, p7591, p7593, p7594};
l6178 = newl;
Ellipse(l6178) = {p7594, p7591, p7594, p7595};
l6179 = newl;
Ellipse(l6179) = {p7595, p7591, p7595, p7592};
ll1544 = newll;
Line Loop(ll1544) = {l6176, l6177, l6178, l6179};
rs1415 = news;
Surface(rs1415) = {ll1544};
p7596 = newp;
Point(p7596) = {-1.0, -1.0, 0.0, 0.06666666666666667};
p7597 = newp;
Point(p7597) = {1.0, -1.0, 0.0, 0.06666666666666667};
p7598 = newp;
Point(p7598) = {1.0, 1.0, 0.0, 0.06666666666666667};
p7599 = newp;
Point(p7599) = {-1.0, 1.0, 0.0, 0.06666666666666667};
l6180 = newl;
Line(l6180) = {p7596, p7597};
l6181 = newl;
Line(l6181) = {p7597, p7598};
l6182 = newl;
Line(l6182) = {p7598, p7599};
l6183 = newl;
Line(l6183) = {p7599, p7596};
ll1545 = newll;
Line Loop(ll1545) = {l6180, l6181, l6182, l6183};
s129 = news;
Plane Surface(s129) = {ll1545,ll1541,ll1542,ll1543,ll1544};
Physical Surface(1) = {s129};
Physical Surface(0) = {rs1412, rs1413, rs1414, rs1415};
Physical Line(2) = {l6180, l6181, l6182, l6183};
Transfinite Line {l6180, l6181, l6182, l6183} = 31;