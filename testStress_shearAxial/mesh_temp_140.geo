// This code was created by pygmsh v6.0.2.
p212 = newp;
Point(p212) = {-0.5, -0.5, 0.0, 0.06666666666666667};
p213 = newp;
Point(p213) = {-0.20601902829352747, -0.5, 0.0, 0.06666666666666667};
p214 = newp;
Point(p214) = {-0.5, -0.20601902829352747, 0.0, 0.06666666666666667};
p215 = newp;
Point(p215) = {-0.7939809717064725, -0.49999999999999994, 0.0, 0.06666666666666667};
p216 = newp;
Point(p216) = {-0.5, -0.7939809717064725, 0.0, 0.06666666666666667};
l172 = newl;
Ellipse(l172) = {p213, p212, p213, p214};
l173 = newl;
Ellipse(l173) = {p214, p212, p214, p215};
l174 = newl;
Ellipse(l174) = {p215, p212, p215, p216};
l175 = newl;
Ellipse(l175) = {p216, p212, p216, p213};
ll43 = newll;
Line Loop(ll43) = {l172, l173, l174, l175};
rs40 = news;
Surface(rs40) = {ll43};
p217 = newp;
Point(p217) = {0.5, -0.5, 0.0, 0.06666666666666667};
p218 = newp;
Point(p218) = {0.7745100946568753, -0.5, 0.0, 0.06666666666666667};
p219 = newp;
Point(p219) = {0.5, -0.22548990534312469, 0.0, 0.06666666666666667};
p220 = newp;
Point(p220) = {0.22548990534312469, -0.49999999999999994, 0.0, 0.06666666666666667};
p221 = newp;
Point(p221) = {0.49999999999999994, -0.7745100946568753, 0.0, 0.06666666666666667};
l176 = newl;
Ellipse(l176) = {p218, p217, p218, p219};
l177 = newl;
Ellipse(l177) = {p219, p217, p219, p220};
l178 = newl;
Ellipse(l178) = {p220, p217, p220, p221};
l179 = newl;
Ellipse(l179) = {p221, p217, p221, p218};
ll44 = newll;
Line Loop(ll44) = {l176, l177, l178, l179};
rs41 = news;
Surface(rs41) = {ll44};
p222 = newp;
Point(p222) = {-0.5, 0.5, 0.0, 0.06666666666666667};
p223 = newp;
Point(p223) = {-0.16265387476426513, 0.5, 0.0, 0.06666666666666667};
p224 = newp;
Point(p224) = {-0.5, 0.8373461252357348, 0.0, 0.06666666666666667};
p225 = newp;
Point(p225) = {-0.8373461252357348, 0.5, 0.0, 0.06666666666666667};
p226 = newp;
Point(p226) = {-0.5000000000000001, 0.16265387476426513, 0.0, 0.06666666666666667};
l180 = newl;
Ellipse(l180) = {p223, p222, p223, p224};
l181 = newl;
Ellipse(l181) = {p224, p222, p224, p225};
l182 = newl;
Ellipse(l182) = {p225, p222, p225, p226};
l183 = newl;
Ellipse(l183) = {p226, p222, p226, p223};
ll45 = newll;
Line Loop(ll45) = {l180, l181, l182, l183};
rs42 = news;
Surface(rs42) = {ll45};
p227 = newp;
Point(p227) = {0.5, 0.5, 0.0, 0.06666666666666667};
p228 = newp;
Point(p228) = {0.7905452745984376, 0.5, 0.0, 0.06666666666666667};
p229 = newp;
Point(p229) = {0.5, 0.7905452745984376, 0.0, 0.06666666666666667};
p230 = newp;
Point(p230) = {0.20945472540156235, 0.5, 0.0, 0.06666666666666667};
p231 = newp;
Point(p231) = {0.49999999999999994, 0.20945472540156235, 0.0, 0.06666666666666667};
l184 = newl;
Ellipse(l184) = {p228, p227, p228, p229};
l185 = newl;
Ellipse(l185) = {p229, p227, p229, p230};
l186 = newl;
Ellipse(l186) = {p230, p227, p230, p231};
l187 = newl;
Ellipse(l187) = {p231, p227, p231, p228};
ll46 = newll;
Line Loop(ll46) = {l184, l185, l186, l187};
rs43 = news;
Surface(rs43) = {ll46};
p232 = newp;
Point(p232) = {-1.0, -1.0, 0.0, 0.06666666666666667};
p233 = newp;
Point(p233) = {1.0, -1.0, 0.0, 0.06666666666666667};
p234 = newp;
Point(p234) = {1.0, 1.0, 0.0, 0.06666666666666667};
p235 = newp;
Point(p235) = {-1.0, 1.0, 0.0, 0.06666666666666667};
l188 = newl;
Line(l188) = {p232, p233};
l189 = newl;
Line(l189) = {p233, p234};
l190 = newl;
Line(l190) = {p234, p235};
l191 = newl;
Line(l191) = {p235, p232};
ll47 = newll;
Line Loop(ll47) = {l188, l189, l190, l191};
s3 = news;
Plane Surface(s3) = {ll47,ll43,ll44,ll45,ll46};
Physical Surface(1) = {s3};
Physical Surface(0) = {rs40, rs41, rs42, rs43};
Physical Line(2) = {l188, l189, l190, l191};
Transfinite Line {l188, l189, l190, l191} = 31;