// This code was created by pygmsh v6.0.2.
p47156 = newp;
Point(p47156) = {-0.5, -0.5, 0.0, 0.06666666666666667};
p47157 = newp;
Point(p47157) = {-0.21904669356005002, -0.5, 0.0, 0.06666666666666667};
p47158 = newp;
Point(p47158) = {-0.5, -0.21904669356005002, 0.0, 0.06666666666666667};
p47159 = newp;
Point(p47159) = {-0.78095330643995, -0.49999999999999994, 0.0, 0.06666666666666667};
p47160 = newp;
Point(p47160) = {-0.5, -0.78095330643995, 0.0, 0.06666666666666667};
l38368 = newl;
Ellipse(l38368) = {p47157, p47156, p47157, p47158};
l38369 = newl;
Ellipse(l38369) = {p47158, p47156, p47158, p47159};
l38370 = newl;
Ellipse(l38370) = {p47159, p47156, p47159, p47160};
l38371 = newl;
Ellipse(l38371) = {p47160, p47156, p47160, p47157};
ll9592 = newll;
Line Loop(ll9592) = {l38368, l38369, l38370, l38371};
rs8788 = news;
Surface(rs8788) = {ll9592};
p47161 = newp;
Point(p47161) = {0.5, -0.5, 0.0, 0.06666666666666667};
p47162 = newp;
Point(p47162) = {0.8606142676746572, -0.5, 0.0, 0.06666666666666667};
p47163 = newp;
Point(p47163) = {0.5, -0.1393857323253428, 0.0, 0.06666666666666667};
p47164 = newp;
Point(p47164) = {0.1393857323253428, -0.49999999999999994, 0.0, 0.06666666666666667};
p47165 = newp;
Point(p47165) = {0.49999999999999994, -0.8606142676746572, 0.0, 0.06666666666666667};
l38372 = newl;
Ellipse(l38372) = {p47162, p47161, p47162, p47163};
l38373 = newl;
Ellipse(l38373) = {p47163, p47161, p47163, p47164};
l38374 = newl;
Ellipse(l38374) = {p47164, p47161, p47164, p47165};
l38375 = newl;
Ellipse(l38375) = {p47165, p47161, p47165, p47162};
ll9593 = newll;
Line Loop(ll9593) = {l38372, l38373, l38374, l38375};
rs8789 = news;
Surface(rs8789) = {ll9593};
p47166 = newp;
Point(p47166) = {-0.5, 0.5, 0.0, 0.06666666666666667};
p47167 = newp;
Point(p47167) = {-0.19502867688282788, 0.5, 0.0, 0.06666666666666667};
p47168 = newp;
Point(p47168) = {-0.5, 0.8049713231171721, 0.0, 0.06666666666666667};
p47169 = newp;
Point(p47169) = {-0.8049713231171721, 0.5, 0.0, 0.06666666666666667};
p47170 = newp;
Point(p47170) = {-0.5000000000000001, 0.19502867688282788, 0.0, 0.06666666666666667};
l38376 = newl;
Ellipse(l38376) = {p47167, p47166, p47167, p47168};
l38377 = newl;
Ellipse(l38377) = {p47168, p47166, p47168, p47169};
l38378 = newl;
Ellipse(l38378) = {p47169, p47166, p47169, p47170};
l38379 = newl;
Ellipse(l38379) = {p47170, p47166, p47170, p47167};
ll9594 = newll;
Line Loop(ll9594) = {l38376, l38377, l38378, l38379};
rs8790 = news;
Surface(rs8790) = {ll9594};
p47171 = newp;
Point(p47171) = {0.5, 0.5, 0.0, 0.06666666666666667};
p47172 = newp;
Point(p47172) = {0.7408623055256796, 0.5, 0.0, 0.06666666666666667};
p47173 = newp;
Point(p47173) = {0.5, 0.7408623055256796, 0.0, 0.06666666666666667};
p47174 = newp;
Point(p47174) = {0.25913769447432045, 0.5, 0.0, 0.06666666666666667};
p47175 = newp;
Point(p47175) = {0.49999999999999994, 0.25913769447432045, 0.0, 0.06666666666666667};
l38380 = newl;
Ellipse(l38380) = {p47172, p47171, p47172, p47173};
l38381 = newl;
Ellipse(l38381) = {p47173, p47171, p47173, p47174};
l38382 = newl;
Ellipse(l38382) = {p47174, p47171, p47174, p47175};
l38383 = newl;
Ellipse(l38383) = {p47175, p47171, p47175, p47172};
ll9595 = newll;
Line Loop(ll9595) = {l38380, l38381, l38382, l38383};
rs8791 = news;
Surface(rs8791) = {ll9595};
p47176 = newp;
Point(p47176) = {-1.0, -1.0, 0.0, 0.06666666666666667};
p47177 = newp;
Point(p47177) = {1.0, -1.0, 0.0, 0.06666666666666667};
p47178 = newp;
Point(p47178) = {1.0, 1.0, 0.0, 0.06666666666666667};
p47179 = newp;
Point(p47179) = {-1.0, 1.0, 0.0, 0.06666666666666667};
l38384 = newl;
Line(l38384) = {p47176, p47177};
l38385 = newl;
Line(l38385) = {p47177, p47178};
l38386 = newl;
Line(l38386) = {p47178, p47179};
l38387 = newl;
Line(l38387) = {p47179, p47176};
ll9596 = newll;
Line Loop(ll9596) = {l38384, l38385, l38386, l38387};
s804 = news;
Plane Surface(s804) = {ll9596,ll9592,ll9593,ll9594,ll9595};
Physical Surface(1) = {s804};
Physical Surface(0) = {rs8788, rs8789, rs8790, rs8791};
Physical Line(2) = {l38384, l38385, l38386, l38387};
Transfinite Line {l38384, l38385, l38386, l38387} = 31;