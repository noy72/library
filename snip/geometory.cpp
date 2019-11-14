#getTangent
# 円と円の共通接線を求める
#shareSegmnet
# 線分の重なり、多角形が接するかを判定
#voronoiCell
# ボロノイ図
#bisectorOfEdge
# 辺の二等分線
#tangentPoint
# 円と点、円と円の接点を求める
#incircle
# 三角形の内接円を求める
#bisectorOfAngle
# 角abcの二等分線を求める
#pointSymmetry
# 多角形が点対象となる点の座標
#dividedPolygonNumber
# 凹多角形を線分lで切断した際の多角形の数
#isConvex
# 凸多角形かどうかの判定
#area
# 面積を求める
#convexCut
# 多角形の切り取り
#diameterOfConvexPolygon
# キャリパー法を用いて凸多角形の直径を求める
#convexHull
# 凸包
#angle
# なす角を求める
#pointInPolygon
# 多角形にある一点が内包されているかを判定
#geometory
# 基本クラスと交差判定など

snippet     geometory
abbr        基本クラスと交差判定など
 typedef complex<double> Point;
 typedef Point Vector;
 struct Segment{					//線分を表す構造体
 	Point p1, p2;
 	Segment(Point p1 = Point(), Point p2 = Point()) : p1(p1), p2(p2) {}
    void out(){
        cout << p1 << ' ' << p2 << endl;
    }
 };
 typedef Segment Line;			//直線を表す構造体
 typedef vector<Point> Polygon;	//多角形を表す構造体
 
 namespace std{
 	bool operator < (const Point &a, const Point &b){
 		return real(a) != real(b) ? real(a) < real(b) : imag(a) < imag(b);
 	}
 	bool operator == (const Point &a, const Point &b){
 		return a.real() == b.real() && a.imag() == b.imag();
 	}
	bool operator == (const Segment &a, const Segment &b){
		return (a.p1 == b.p1 and a.p2 == b.p2) or (a.p1 == b.p2 and a.p2 == b.p1);
	}
 }
 
 class Circle{
 	public:
 		Point p;
 		double r;
 		Circle(Point p = Point(), double r = 0.0): p(p), r(r) {}
 };
 
 // 許容する誤差
 #define EPS (1e-10)
 #define INF (1e10)
 
 // ベクトルaの絶対値を求める
 //double length = abs(a);
 
 // 2点a,b間の距離を求める
 //double distance = abs(a-b);
 
 /*
 // ベクトルaの単位ベクトルを求める
 Point b = a / abs(a);
 
 // ベクトルaの法線ベクトルn1,n2を求める
 Point n1 = a * Point(0, 1);
 Point n2 = a * Point(0, -1);
 */
 
 int ccw(Point, Point, Point);
 
 // 2つのスカラーが等しいかどうか
 bool EQ(double a, double b){
 	return (abs(a - b) < EPS);
 }
 
 // 2つのベクトルが等しいかどうか
 bool EQV(Vector a, Vector b){
 	return ( EQ(a.real(), b.real()) && EQ(a.imag(), b.imag()) );
 }
 
 // 内積 (dot product) : a・b = |a||b|cosΘ
 double dot(Point a, Point b) {
 	return (a.real() * b.real() + a.imag() * b.imag());
 }
 
 // 外積 (cross product) : a×b = |a||b|sinΘ
 double cross(Point a, Point b) {
 	return (a.real() * b.imag() - a.imag() * b.real());
 }
 
 // 2直線の直交判定 : a⊥b <=> dot(a, b) = 0
 bool isOrthogonal(Point a1, Point a2, Point b1, Point b2) {
 	return EQ( dot(a1-a2, b1-b2), 0.0 );
 }
 bool isOrthogonal(Line s1, Line s2) { return isOrthogonal(s1.p1, s1.p2, s2.p1, s2.p2); }
 
 // 2直線の平行判定 : a//b <=> cross(a, b) = 0
 bool isParallel(Point a1, Point a2, Point b1, Point b2) {
 	return EQ( cross(a1-a2, b1-b2), 0.0 );
 }
 bool isParallel(Line s1, Line s2) { return isParallel(s1.p1, s1.p2, s2.p1, s2.p2); }
 
 // 点cが直線a,b上にあるかないか
 bool isPointOnLine(Point a, Point b, Point c) {
 	return EQ( cross(b-a, c-a), 0.0 );
 }
 bool isPointOnLine(Line s, Point c) { return isPointOnLine(s.p1, s.p2, c); }
 
 // 点a,bを通る直線と点cとの距離
 double distanceLPoint(Point a, Point b, Point c) {
 	return abs(cross(b-a, c-a)) / abs(b-a);
 }
 double distanceLPoint(Line s, Point c) { return distanceLPoint(s.p1, s.p2, c); }
 
 // 点a,bを端点とする線分と点cとの距離
 double distanceLsPoint(Point a, Point b, Point c) {
 	if ( dot(b-a, c-a) < EPS ) return abs(c-a);
 	if ( dot(a-b, c-b) < EPS ) return abs(c-b);
 	return abs(cross(b-a, c-a)) / abs(b-a);
 }
 double distanceLsPoint(Segment s, Point c) { return distanceLsPoint(s.p1, s.p2, c); }
 
 // a1,a2を端点とする線分とb1,b2を端点とする線分の交差判定
 // 端点が重なる場合も、線分が交差しているとみなす
 bool isIntersectedLs(Point a1, Point a2, Point b1, Point b2) {
 	return ( ccw(a1, a2, b1) * ccw(a1, a2, b2) <= 0 &&
 			ccw(b1, b2, a1) * ccw(b1, b2, a2) <= 0 );
 }
 bool isIntersectedLs(Segment s1, Segment s2) { return isIntersectedLs(s1.p1, s1.p2, s2.p1, s2.p2); }
 
 // 端点が重なっているかを検出する
 bool isContainSamePoints(Segment s1, Segment s2){
 	if(abs(s1.p1 - s2.p1) < EPS) return true;
 	if(abs(s1.p1 - s2.p2) < EPS) return true;
 	if(abs(s1.p2 - s2.p1) < EPS) return true;
 	if(abs(s1.p2 - s2.p2) < EPS) return true;
 	return false;
 }
 
 // a1,a2を端点とする線分とb1,b2を端点とする線分の交点計算
 Point intersectionLs(Point a1, Point a2, Point b1, Point b2) {
 	Vector base = b2 - b1;
 	double d1 = abs(cross(base, a1 - b1));
 	double d2 = abs(cross(base, a2 - b1));
 	double t = d1 / (d1 + d2);
 
 	return Point(a1 + (a2 - a1) * t);
 }
 Point intersectionLs(Segment s1, Segment s2) { return intersectionLs(s1.p1, s1.p2, s2.p1, s2.p2); }
 
 // a1,a2を通る直線とb1,b2を通る直線の交差判定
 bool isIntersectedL(Point a1, Point a2, Point b1, Point b2) {
 	return !EQ( cross(a1-a2, b1-b2), 0.0 );
 }
 bool isIntersectedL(Line l1, Line l2) { return isIntersectedL(l1.p1, l1.p2, l2.p1, l2.p2); }
 
 // a1,a2を通る直線とb1,b2を通る直線の交点計算
 Point intersectionL(Point a1, Point a2, Point b1, Point b2) {
 	Point a = a2 - a1; Point b = b2 - b1;
 	return a1 + a * cross(b, b1-a1) / cross(b, a);
 }
 Point intersectionL(Line l1, Line l2) { return intersectionL(l1.p1, l1.p2, l2.p1, l2.p2); }
 
 // 線分s1と線分s2の距離
 double distanceLL(Segment s1, Segment s2){
 	if(isIntersectedLs(s1.p1, s1.p2, s2.p1, s2.p2) ) return 0.0;
 	return min(
 			min(distanceLsPoint(s1.p1, s1.p2, s2.p1),
 				distanceLsPoint(s1.p1, s1.p2, s2.p2)),
 			min(distanceLsPoint(s2.p1, s2.p2, s1.p1),
 				distanceLsPoint(s2.p1, s2.p2, s1.p2)) );
 }
 double distanceLL(Point p0, Point p1, Point p2, Point p3){
 	Segment s1 = Segment{p0, p1}, s2 = Segment{p2, p3};
 	return distanceLL(s1, s2);
 }
 
 // 線分sに対する点pの射影
 Point project(Segment s, Point p){
 	Vector base = s.p2 - s.p1;
 	double r = dot(p - s.p1, base) / norm(base);
 	return Point(s.p1 + base * r);
 }
 
 //線分sを対象軸とした点pの線対称の点
 Point reflect(Segment s, Point p){
 	return Point(p + (project(s, p) - p) * 2.0);
 }
 
 //点pをangle分だけ時計回りに回転
 Point rotation(Point p, double angle){
 	double x, y;
 	x = p.real() * cos(angle) - p.imag() * sin(angle);
 	y = p.real() * sin(angle) + p.imag() * cos(angle);
 	return Point(x, y);
 }
 
 //円cと線分lの交点
 pair<Point, Point> getCrossPoints(Circle c, Line l){
 	Vector pr = project(l, c.p);
 	Vector e = (l.p2 - l.p1) / abs(l.p2 - l.p1);
 	double base = sqrt(c.r * c.r - norm(pr - c.p));
 	return make_pair(pr + e * base, pr - e * base);
 }
 
 //円c1と円c2の交点
 double arg(Vector p) { return atan2(p.imag(), p.real()); }
 Vector polar(double a, double r) { return Point(cos(r) * a, sin(r) *a); }
 
 pair<Point, Point> getCrossPoints(Circle c1, Circle c2){
 	double d = abs(c1.p - c2.p);
 	double a = acos((c1.r * c1.r + d * d - c2.r * c2.r) / (2 * c1.r * d));
 	double t = arg(c2.p - c1.p);
 	return make_pair(Point(c1.p + polar(c1.r, t + a)), Point(c1.p + polar(c1.r, t - a)));
 }

 static const int COUNTER_CLOCKWISE = 1;
 static const int CLOCKWISE = -1;
 static const int ONLINE_BACK = 2;
 static const int ONLINE_FRONT = -2;
 static const int ON_SEGMENT = 0;
 
 int ccw(Point p0, Point p1, Point p2){
 	Vector a = p1 - p0;
 	Vector b = p2 - p0;
 	if( cross(a, b) > EPS ) return COUNTER_CLOCKWISE;
 	if( cross(a, b) < -EPS ) return CLOCKWISE;
 	if( dot(a, b) < -EPS ) return ONLINE_BACK;
 	if( abs(a) < abs(b) ) return ONLINE_FRONT;
 
 	return ON_SEGMENT;
 }

snippet     pointInPolygon
abbr        多角形にある一点が内包されているかを判定
 //点の内包
 static const int IN = 2;
 static const int ON = 1;
 static const int OUT = 0;
 
 int contains(Polygon g, Point p){
 	int n = g.size();
 	bool x = false;
 	rep(i,n){
 		Point a = g[i] - p, b = g[(i + 1) % n] - p;
 		if( abs(cross(a, b)) < EPS && dot(a,  b) < EPS ) return ON;
 		if( a.imag() > b.imag() ) swap(a, b);
 		if( a.imag() < EPS && EPS < b.imag() && cross(a, b) > EPS ) x = not x;
 	}
 	return ( x ? IN : OUT );
 }

snippet     angle
abbr        なす角を求める
 //弧度法から度数法の変換
 double radianToDegree(double rad){
 	return 180 * rad / M_PI;
 }
 
 //度数法から変弧度法の換
 double degreeToRadian(double deg){
 	return M_PI * deg / 180;
 }
 
 //2つのベクトルからなる角度を求める
 double angleOf2Vector(Vector a, Vector b){
 	return acos( dot(a,b) / (abs(a) * abs(b)) );
 }

snippet     convexHull
abbr        凸包
 Polygon convexHull( Polygon s ){
 	Polygon u;
 	if( s.size() < 3 ) return s;
 	sort(s.begin(), s.end());
 
 	range(i,0,s.size()){
 		//== COUNTER_CLOCKWISEだと内角は180以下（一直線上に並んでいても、頂点として数える）
 		//!= CLOCKWISEだと内角は180未満（一直線上の頂点は数えない）
 		for(int n = u.size(); n >= 2 && ccw(u[n-2], u[n-1], s[i]) == COUNTER_CLOCKWISE; n--){
 			u.pop_back();
 		}
 		u.emplace_back(s[i]);
 	}
 
 	for(int i = s.size() - 2; i >= 0; i--){
 		//ここも == と != を変更する
 		for(int n = u.size(); n >= 2 && ccw(u[n-2], u[n-1], s[i]) == COUNTER_CLOCKWISE; n--){
 			u.pop_back();
 		}
 		u.emplace_back(s[i]);
 	}
 
 	reverse(u.begin(), u.end());
 	u.pop_back();
 
 	//最も下にある点の中で最も右にある点から反時計回りに並び替え
 	/*
 	   int i = 0;
 	   while(i < u.size() - 1){
 	   if(u[i].imag() > u[i + 1].imag()){
 	   u.emplace_back(u[i]);
 	   u.erase(u.begin());
 	   continue;
 	   }else if(u[i].imag() == u[i + 1].imag() && u[i].real() > u[i + 1].real()){
 	   u.emplace_back(u[i]);
 	   u.erase(u.begin());
 	   continue;
 	   }
 	   break;
 	   }
 	   */
 
 	return u;
 }

snippet     diameterOfConvexPolygon
abbr        キャリパー法を用いて凸多角形の直径を求める
 double diameterOfConvexPolygon(Polygon p){
 	Polygon s = convexHull(p);
 	int n = s.size();
 
 	if(n == 2) return abs(s[1] - s[0]);
 
 	int i = 0, j = 0;
 	rep(k,n){
 		if(not (s[i] < s[k])) i = k;
 		if(s[j] < s[k]) j = k;
 	}
 
 	double ret = 0.0;
 	int is = i, js = j;
 
 	while(i != js || j != is){
 		ret = max(ret, abs(s[i] - s[j]));
 		if(cross(s[(i + 1) % n] - s[i], s[(j + 1) % n] - s[j]) < 0){
 			i = (i + 1) % n;
 		}else{
 			j = (j + 1) % n;
 		}
 	}
 	return ret;
 }

snippet     convexCut
abbr        多角形の切り取り
 Point getCrossPointLL(Line a, Line b){
 	double A = cross(a.p2 - a.p1, b.p2 - b.p1);
 	double B = cross(a.p2 - a.p1, a.p2 - b.p1);
 	if(abs(A) < EPS && abs(B) < EPS) return b.p1;
 	return b.p1 + (b.p2 - b.p1) * (B / A);
 }
 
 Polygon convexCut(Polygon p, Line l) {
 	Polygon q;
 	rep(i,p.size()){
 		Point a = p[i], b = p[(i + 1) % p.size()];
 		if (ccw(l.p1, l.p2, a) != -1) q.emplace_back(a);
 		if (ccw(l.p1, l.p2, a) * ccw(l.p1, l.p2, b) < 0){
 			q.emplace_back(getCrossPointLL(Line{a, b}, l));
 		}
 	}
 	return q;
 }

snippet     area
abbr        面積を求める
 //三角形の面積
 double areaOfTriangle(Point a, Point b, Point c){
 	double w, x, y, z;
 	w = b.real()-a.real();
 	x = b.imag()-a.imag();
 	y = c.real()-a.real();
 	z = c.imag()-a.imag();
 	return abs((w * z - x * y) / 2);
 }
 
 //多角形の面積
 double areaOfPolygon(Polygon g){
 	int n = g.size();
 	double ret = 0.0;
 	rep(i,n) ret += cross(g[i], g[ (i + 1) % n ]);
 return abs(ret) / 2.0;
 }

snippet     isConvex
abbr        凸多角形かどうかの判定
 bool isConvex(Polygon g){
 	int n = g.size();
 	rep(i,n){
 		if(ccw(g[i], g[(i + 1) % n], g[(i + 2) % n]) == CLOCKWISE) return false;
 	}
 	return true;
 }

snippet     dividedPolygonNumber
abbr        凹多角形を線分lで切断した際の多角形の数
 int dividedPolygonNumber(Polygon p, Line l){
 	int cnt = 0;
 	rep(i,p.size()){
 		if(isIntersectedLs(p[i], p[(i + 1) % p.size()], l.p1, l.p2)) cnt++;
 	}
 	return cnt / 2 + 1;
 }

snippet     pointSymmetry
abbr        多角形が点対象となる点の座標
 Point pointSymmetry(Polygon g){
 	int size = g.size() / 2;
 	if(g.size() % 2) return Point{INF,INF};
 
 	set<Point> s;
 	rep(i,size){
 		rep(j,size){
 			if(i == j) continue;
 			s.insert(intersectionLs(g[i], g[i + size], g[j], g[j + size]));
 		}
 	}
 	if(s.size() > 1) return Point{INF,INF};
 	return *s.begin();
 }

snippet     bisectorOfAngle
abbr        角abcの二等分線を求める
 //2つのベクトルからなる角度を求める
 double angleOf2Vector(Vector a, Vector b){
 	return acos( dot(a,b) / (abs(a) * abs(b)) );
 }

 Line bisectorOfAngle(Point a, Point b, Point c){
 	a-=b;
 	c-=b;
 	//cout << a << ' ' << c << endl;
 
 	double angle = angleOf2Vector(a, c);
 	angle /= 2;
 
 	if(ccw(Point{0,0}, a, c) == COUNTER_CLOCKWISE){
 		return Line{b, rotation(a, angle) + b};
 	}else{
 		return Line{b, rotation(a, -angle) + b};
 	}
 	assert(false);
 }

snippet     incircle
abbr        三角形の内接円を求める
 Circle incircle(Point p[3]){
 	Line bisector1 = requireBisectorOfAngle(p[0], p[1], p[2]);
 	Line bisector2 = requireBisectorOfAngle(p[0], p[2], p[1]);
 	//cout << bisector1.p1 << ' ' << bisector1.p2 << endl;
 	//cout << bisector2.p1 << ' ' << bisector2.p2 << endl;
 	Point center = intersectionL(bisector1, bisector2);
 	//show(center)
 	double r = distanceLPoint(p[0], p[1], center);
 	//show(r)
 	return Circle{center, r};
 }

snippet     tangentPoint
abbr        円と点、円と円の接点を求める
 // 円と点の接線を求める
 pair<Point,Point> getTangentPointointCp(Circle c, Point p) {
 	auto pow = [](double a) -> double {return a * a; };
 	double x = p.real() - c.p.real(), y = p.imag() - c.p.imag();
 
 	double sqt = sqrt((pow(x) + pow(y)) - pow(c.r));
 	double denominator = pow(x) + pow(y);
 
 	double x1 = c.r * (x * c.r + y * sqt) / denominator + c.p.real();
 	double y1 = c.r * (y * c.r - x * sqt) / denominator + c.p.imag();
 	double x2 = c.r * (x * c.r - y * sqt) / denominator + c.p.real();
 	double y2 = c.r * (y * c.r + x * sqt) / denominator + c.p.imag();
 
 	return make_pair(Point(x1, y1), Point(x2, y2));
 }
 
 // 円と円の接線を求める
 vector<Point> getTangentPointointCc(Circle a, Circle b) {
 	auto pow = [](double a) -> double {return a * a; };
 
 	Circle c(Point(b.p.real() - a.p.real(), b.p.imag() - a.p.imag()), b.r);
 
 	vector<Point> res;
 
 	auto f = [&](double R) {
 		double left = a.r + R;
 		double right = sqrt((pow(c.p.real()) + pow(c.p.imag())) - pow(a.r + R));
 		double denominator = pow(c.p.real()) + pow(c.p.imag());
 
 		double x1 = a.r * (c.p.real() * left + c.p.imag() * right) / denominator + a.p.real();
 		double y1 = a.r * (c.p.imag() * left - c.p.real() * right) / denominator + a.p.imag();
 
 		double x2 = a.r * (c.p.real() * left - c.p.imag() * right) / denominator + a.p.real();
 		double y2 = a.r * (c.p.imag() * left + c.p.real() * right) / denominator + a.p.imag();
 
 		pair<Point,Point> p = make_pair(Point(x1, y1), Point(x2, y2));
 		if (not std::isnan(p.first.real())) res.emplace_back(p.first);
 		if (not std::isnan(p.second.real())) res.emplace_back(p.second);
 	};
 
 	f(c.r);		// 内接線
 	f(-c.r);	// 外接線
 
 	return res;
 }

snippet     bisectorOfEdge
abbr        辺の二等分線
 Line bisector(Point a, Point b) {
 	Point A = (a + b) * Point(0.5, 0);
 	return Line(A, A + (b - a) * Point(0, M_PI / 2));
 }

snippet     voronoiCell
abbr        ボロノイ図
 Point getCrossPointLL(Line a, Line b){
 	double A = cross(a.p2 - a.p1, b.p2 - b.p1);
 	double B = cross(a.p2 - a.p1, a.p2 - b.p1);
 	if(abs(A) < EPS && abs(B) < EPS) return b.p1;
 	return b.p1 + (b.p2 - b.p1) * (B / A);
 }
 
 Polygon convexCut(Polygon p, Line l) {
 	Polygon q;
 	rep(i,p.size()){
 		Point a = p[i], b = p[(i + 1) % p.size()];
 		if (ccw(l.p1, l.p2, a) != -1) q.emplace_back(a);
 		if (ccw(l.p1, l.p2, a) * ccw(l.p1, l.p2, b) < 0){
 			q.emplace_back(getCrossPointLL(Line{a, b}, l));
 		}
 	}
 	return q;
 }
 
 Line bisector(Point a, Point b) {
 	Point A = (a + b) * Point(0.5, 0);
 	return Line(A, A + (b - a) * Point(0, M_PI / 2));
 }
 
 // 多角形, 母点（分割するための点）、母点のインデックス -> 母点 s が含まれるボロノイ領域
 Polygon voronoiCell(Polygon g, vector<Point> v, int s) {
 	rep(i, v.size()){
 		if (i != s) g = convexCut(g, bisector(v[s], v[i]));
 	}
 	return g;
 }

snippet     shareSegmnet
abbr        線分の重なり、多角形が接するかを判定
 Point nxt(Polygon& a, int i){
 	return a[(i + 1) % a.size()];
 }
 
 // 線分の重なり判定
 // 線分が重なっているとき true.  それ以外 false. 頂点で重なるときは false
 bool contain(const Segment& s, const Point& p) {
     if (s.p1 == p) return 0;
     if (s.p2 == p) return 0;
     return ccw(s.p1, s.p2, p) == ON_SEGMENT;
 }
 
 bool isSharedLs(Segment s1, Segment s2){
 	auto f = [](Segment s1, Segment s2) -> bool{
 		if(isParallel(s1, s2)){
 			if(contain(s1, s2.p1) or contain(s1, s2.p2)) return true;
 		}
 		return false;
 	};
 	return f(s1, s2) or f(s2, s1) or s1 == s2;
 }
 
 // 二つの多角形が接しているかを判定
 bool shareBorder(Polygon& a, Polygon& b){
 	rep(i,a.size()){
 		rep(j,b.size()){
 			Segment s1 = Segment{a[i], nxt(a,i)}, s2 = Segment{b[j], nxt(b,j)};
 			if(isSharedLs(s1, s2)) return true;
 		}
 	}
 	return false;
 }

snippet     getTangent
abbr        円と円の共通接線を求める
 Point turn(Point p, double t) {
     return p * exp(Point(.0, t * M_PI / 180.0));
 }
 
 vector<Line> tangentCircleCircle(Circle a, Circle b) {
     if (a.r < b.r) swap(a, b);
     double d = abs(a.p - b.p);
     vector<Line> l;
     if (d < EPS) return l;
     if (a.r + b.r < d - EPS) { //hanareteiru
         double t = acos((a.r + b.r) / d);
         t = t * 180 / M_PI;
         l.push_back(Line(a.p + turn(a.r / d * (b.p - a.p), t), b.p + turn(b.r / d * (a.p - b.p), t)));
         l.push_back(Line(a.p + turn(a.r / d * (b.p - a.p), -t), b.p + turn(b.r / d * (a.p - b.p), -t)));
     } else if (a.r + b.r < d + EPS) { //kuttuiteiru soto
         Point p = a.p + a.r / d * (b.p - a.p);
         l.push_back(Line(p, p + turn(b.p - a.p, 90)));
     }
     if (abs(a.r - b.r) < d - EPS) { //majiwatteiru
         double t1 = acos((a.r - b.r) / d);
         t1 = t1 * 180 / M_PI;
         double t2 = 180 - t1;
         l.push_back(Line(a.p + turn(a.r / d * (b.p - a.p), t1), b.p + turn(b.r / d * (a.p - b.p), -t2)));
         l.push_back(Line(a.p + turn(a.r / d * (b.p - a.p), -t1), b.p + turn(b.r / d * (a.p - b.p), t2)));
     } else if (abs(a.r - b.r) < d + EPS) {//kuttuiteiru uti
         Point p = a.p + a.r / d * (b.p - a.p);
         l.push_back(Line(p, p + turn(b.p - a.p, 90)));
     }
     return l;
 }
