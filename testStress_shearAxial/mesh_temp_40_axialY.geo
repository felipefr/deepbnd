// This code was created by pygmsh v6.0.2.
p42188 = newp;
Point(p42188) = {-0.5, -0.5, 0.0, 0.06666666666666667};
p42189 = newp;
Point(p42189) = {-0.23838488381881867, -0.5, 0.0, 0.06666666666666667};
p42190 = newp;
Point(p42190) = {-0.5, -0.23838488381881867, 0.0, 0.06666666666666667};
p42191 = newp;
Point(p42191) = {-0.7616151161811813, -0.49999999999999994, 0.0, 0.06666666666666667};
p42192 = newp;
Point(p42192) = {-0.5, -0.7616151161811813, 0.0, 0.06666666666666667};
l34228 = newl;
Ellipse(l34228) = {p42189, p42188, p42189, p42190};
l34229 = newl;
Ellipse(l34229) = {p42190, p42188, p42190, p42191};
l34230 = newl;
Ellipse(l34230) = {p42191, p42188, p42191, p42192};
l34231 = newl;
Ellipse(l34231) = {p42192, p42188, p42192, p42189};
ll8557 = newll;
Line Loop(ll8557) = {l34228, l34229, l34230, l34231};
rs7960 = news;
Surface(rs7960) = {ll8557};
p42193 = newp;
Point(p42193) = {0.5, -0.5, 0.0, 0.06666666666666667};
p42194 = newp;
Point(p42194) = {0.7475799542439852, -0.5, 0.0, 0.06666666666666667};
p42195 = newp;
Point(p42195) = {0.5, -0.25242004575601484, 0.0, 0.06666666666666667};
p42196 = newp;
Point(p42196) = {0.25242004575601484, -0.49999999999999994, 0.0, 0.06666666666666667};
p42197 = newp;
Point(p42197) = {0.49999999999999994, -0.7475799542439852, 0.0, 0.06666666666666667};
l34232 = newl;
Ellipse(l34232) = {p42194, p42193, p42194, p42195};
l34233 = newl;
Ellipse(l34233) = {p42195, p42193, p42195, p42196};
l34234 = newl;
Ellipse(l34234) = {p42196, p42193, p42196, p42197};
l34235 = newl;
Ellipse(l34235) = {p42197, p42193, p42197, p42194};
ll8558 = newll;
Line Loop(ll8558) = {l34232, l34233, l34234, l34235};
rs7961 = news;
Surface(rs7961) = {ll8558};
p42198 = newp;
Point(p42198) = {-0.5, 0.5, 0.0, 0.06666666666666667};
p42199 = newp;
Point(p42199) = {-0.16651854838418195, 0.5, 0.0, 0.06666666666666667};
p42200 = newp;
Point(p42200) = {-0.5, 0.833481451615818, 0.0, 0.06666666666666667};
p42201 = newp;
Point(p42201) = {-0.833481451615818, 0.5, 0.0, 0.06666666666666667};
p42202 = newp;
Point(p42202) = {-0.5000000000000001, 0.16651854838418195, 0.0, 0.06666666666666667};
l34236 = newl;
Ellipse(l34236) = {p42199, p42198, p42199, p42200};
l34237 = newl;
Ellipse(l34237) = {p42200, p42198, p42200, p42201};
l34238 = newl;
Ellipse(l34238) = {p42201, p42198, p42201, p42202};
l34239 = newl;
Ellipse(l34239) = {p42202, p42198, p42202, p42199};
ll8559 = newll;
Line Loop(ll8559) = {l34236, l34237, l34238, l34239};
rs7962 = news;
Surface(rs7962) = {ll8559};
p42203 = newp;
Point(p42203) = {0.5, 0.5, 0.0, 0.06666666666666667};
p42204 = newp;
Point(p42204) = {0.8450382402970926, 0.5, 0.0, 0.06666666666666667};
p42205 = newp;
Point(p42205) = {0.5, 0.8450382402970926, 0.0, 0.06666666666666667};
p42206 = newp;
Point(p42206) = {0.15496175970290743, 0.5, 0.0, 0.06666666666666667};
p42207 = newp;
Point(p42207) = {0.49999999999999994, 0.15496175970290743, 0.0, 0.06666666666666667};
l34240 = newl;
Ellipse(l34240) = {p42204, p42203, p42204, p42205};
l34241 = newl;
Ellipse(l34241) = {p42205, p42203, p42205, p42206};
l34242 = newl;
Ellipse(l34242) = {p42206, p42203, p42206, p42207};
l34243 = newl;
Ellipse(l34243) = {p42207, p42203, p42207, p42204};
ll8560 = newll;
Line Loop(ll8560) = {l34240, l34241, l34242, l34243};
rs7963 = news;
Surface(rs7963) = {ll8560};
p42208 = newp;
Point(p42208) = {-1.0, -1.0, 0.0, 0.06666666666666667};
p42209 = newp;
Point(p42209) = {1.0, -1.0, 0.0, 0.06666666666666667};
p42210 = newp;
Point(p42210) = {1.0, 1.0, 0.0, 0.06666666666666667};
p42211 = newp;
Point(p42211) = {-1.0, 1.0, 0.0, 0.06666666666666667};
l34244 = newl;
Line(l34244) = {p42208, p42209};
l34245 = newl;
Line(l34245) = {p42209, p42210};
l34246 = newl;
Line(l34246) = {p42210, p42211};
l34247 = newl;
Line(l34247) = {p42211, p42208};
ll8561 = newll;
Line Loop(ll8561) = {l34244, l34245, l34246, l34247};
s597 = news;
Plane Surface(s597) = {ll8561,ll8557,ll8558,ll8559,ll8560};
Physical Surface(1) = {s597};
Physical Surface(0) = {rs7960, rs7961, rs7962, rs7963};
Physical Line(2) = {l34244, l34245, l34246, l34247};
Transfinite Line {l34244, l34245, l34246, l34247} = 31;