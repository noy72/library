snippet     dice
abbr        正六面体
    enum FACE { TOP, BOTTOM, FRONT, BACK, LEFT, RIGHT };
    template <class T>
    class dice {
    public:
      dice() {
        id[TOP] = 0; id[FRONT] = 1; id[LEFT] = 2;
        id[RIGHT] = 3; id[BACK] = 4; id[BOTTOM] = 5;
      }
      T& operator[] (FACE f) { return var[id[f]]; }
      const T& operator[] (FACE f) const { return var[id[f]]; }
      bool operator==(const dice<T>& b) const {
        const dice<T> &a = *this;
        return a[TOP] == b[TOP] && a[BOTTOM] == b[BOTTOM] &&
               a[FRONT] == b[FRONT] && a[BACK] == b[BACK] &&
               a[LEFT] == b[LEFT] && a[RIGHT] == b[RIGHT];
      }
      void roll_x() { roll(TOP, BACK, BOTTOM, FRONT); }
      void roll_y() { roll(TOP, LEFT, BOTTOM, RIGHT); }
      void roll_z() { roll(FRONT, RIGHT, BACK, LEFT); }
      vector<dice> all_rolls() {
        vector<dice> ret;
        for (int k = 0; k < 6; (k&1?roll_y():roll_x()),++k)
          for (int i = 0; i < 4; roll_z(), ++i)
            ret.push_back(*this);
        return ret;
      }
      bool equivalent_to(const dice& di) {
        for (int k = 0; k < 6; (k&1?roll_y():roll_x()),++k)
          for (int i = 0; i < 4; roll_z(), ++i)
            if (*this == di) return true;
        return false;
      }
    private:
      void roll(FACE a, FACE b, FACE c, FACE d) {
        T tmp = id[a];
        id[a] = id[b]; id[b] = id[c];
        id[c] = id[d]; id[d] = tmp;
      }
      T var[6];
      int id[6];
    };

snippet     geometory3D
abbr        空間幾何学

    const double EPS = 1e-9;
    auto equals = [](double a, double b) { return fabs(a - b) < EPS; };

    struct Point {
        double x, y, z;
        Point() {}
        Point(double x, double y, double z) : x(x), y(y), z(z) {}
        Point operator+(Point p) { return Point(x + p.x, y + p.y, z + p.z); }
        Point operator-(Point p) { return Point(x - p.x, y - p.y, z - p.z); }
        Point operator*(double k) { return Point(x * k, y * k, z * k); }
        Point operator/(double k) { return Point(x / k, y / k, z / k); }
        Point operator*(Point p) {
            return Point(y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x);
        }
        double operator^(Point p) { return x * p.x + y * p.y + z * p.z; }
        double norm() { return x * x + y * y + z * z; }
        double abs() { return sqrt(norm()); }
        bool operator<(const Point &p) const {
            if (x != p.x) return x < p.x;
            if (y != p.y) return y < p.y;
            return z < p.z;
        }
        bool operator==(const Point &p) const {
            return fabs(x - p.x) < EPS && fabs(y - p.y) < EPS &&
                   fabs(z - p.z) < EPS;
        }
    };
    istream &operator>>(istream &is, Point &p) {
        is >> p.x >> p.y >> p.z;
        return is;
    }
    ostream &operator<<(ostream &os, Point p) {
        os << fixed << setprecision(12) << p.x << " " << p.y << " " << p.z;
        return os;
    }

    typedef Point Vector;
    typedef vector<Point> Polygon;

    struct Segment {
        Point p1, p2;
        Segment() {}
        Segment(Point p1, Point p2) : p1(p1), p2(p2) {}
    };
    typedef Segment Line;

    istream &operator>>(istream &is, Segment &s) {
        is >> s.p1 >> s.p2;
        return is;
    }

    struct Sphere {
        Point c;
        double r;
        Sphere() {}
        Sphere(Point c, double r) : c(c), r(r) {}
    };

    istream &operator>>(istream &is, Sphere &c) {
        is >> c.c >> c.r;
        return is;
    }

    double norm(Vector a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
    double abs(Vector a) { return sqrt(norm(a)); }
    double dot(Vector a, Vector b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    Vector cross(Vector a, Vector b) {
        return Vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
    }

    Point project(Line l, Point p) {
        Point b = l.p2 - l.p1;
        double t = dot(p - l.p1, b) / norm(b);
        return l.p1 + b * t;
    }

    Point reflect(Line l, Point p) { return p + (project(l, p) - p) * 2.0; }

    double getDistanceLP(Line l, Point p) {
        return abs(cross(l.p2 - l.p1, p - l.p1) / abs(l.p2 - l.p1));
    }

    double getDistanceSP(Segment s, Point p) {
        if (dot(s.p2 - s.p1, p - s.p1) < 0.0) return abs(p - s.p1);
        if (dot(s.p1 - s.p2, p - s.p2) < 0.0) return abs(p - s.p2);
        return getDistanceLP(s, p);
    }

    bool intersectSC(Segment s, Sphere c) {
        double d = getDistanceSP(s, c.c);
        if (d > c.r) return 0;
        return !((abs(s.p1 - c.c) <= c.r) && (abs(s.p2 - c.c) <= c.r));
    }
    bool intersectSC(Point p1, Point p2, Sphere c) {
        return intersectSC(Segment(p1, p2), c);
    }


snippet     bfs_list
abbr        グラフのBFS
options     head
 vector<int> bfs(int s, const Graph& g) {
 	queue<int> q;
 	q.emplace(s);
  
 	vector<int> dis(g.size(), 1e18);
 	dis[s] = 0;
  
 	vector<int> pre(g.size(), -1);
 	while (not q.empty()) {
 		int cur = q.front();
 		q.pop();
 		for (auto to : g[cur]) {
 			if (dis[to] != 1e18) continue;
 			pre[to] = cur;
 			dis[to] = dis[cur] + 1;
 			q.emplace(to);
 		}
 	}
 	return dis;
 }
snippet     bipartite_graph
abbr        $BFs?':L?'(B
 void dfs(vector<vector<int>>& g, int cur, vector<int>& color){
 	for(auto to : g[cur]){
 		if(color[to] != -1) continue;
 		color[to] = color[cur] ^ 1;
 		dfs(g, to, color);
 	}
 }
 
 // $BFsIt%0%i%U$+$r%A%'%C%/(B
 bool check(vector<vector<int>>& g, vector<int>& color){
 	rep(i,g.size()){
 		for(auto to : g[i]){
 			if(color[i] == color[to]){
 				return false;
 			}
 		}
 	}
 	return true;
 }
snippet     gabowEdmonds
abbr        一般マッチング
    vector<pair<int, int>> gabowEdmonds(const vector<vector<int>>& graph) {
        int n = graph.size();
        vector<vector<pair<int, int>>> g(n + 1);
        vector<pair<int, int>> edges;
        int cnt = n + 1;
        for (int i = 0; i < n; i++) {
            for (auto to : graph[i]) {
                if (i < to) {
                    g[to + 1].emplace_back(i + 1, cnt);
                    g[i + 1].emplace_back(to + 1, cnt++);
                    edges.emplace_back(i + 1, to + 1);
                }
            }
        }

        vector<int> mate(n + 1, 0), label(n + 1, -1), first(n + 1, 0);
        queue<int> q;

        // first$B$NCY1dI>2A(B
        function<int(int)> eval_first = [&](int x) {
            if (label[first[x]] < 0) return first[x];
            first[x] = eval_first(first[x]);
            return first[x];
        };

        function<void(int, int)> rematch = [&](int v, int w) {
            int t = mate[v];
            mate[v] = w;
            if (mate[t] != v) return;

            if (label[v] <= n) {
                mate[t] = label[v];
                rematch(label[v], t);
            } else {
                int x, y;
                tie(x, y) = edges[label[v] - n - 1];
                rematch(x, y);
                rematch(y, x);
            }
        };

        function<void(int, int, int)> assignLabel = [&](int x, int y, int num) {
            int r = eval_first(x);
            int s = eval_first(y);
            if (r == s) return;

            label[r] = -num;
            label[s] = -num;
            int join = 0;
            while (true) {
                if (s != 0) swap(r, s);
                r = eval_first(label[mate[r]]);
                if (label[r] == -num) {
                    join = r;
                    break;
                }
                label[r] = -num;
            }

            int v = first[x];
            while (v != join) {
                q.emplace(v);
                label[v] = num;
                first[v] = join;
                v = first[label[mate[v]]];
            }
            v = first[y];
            while (v != join) {
                q.emplace(v);
                label[v] = num;
                first[v] = join;
                v = first[label[mate[v]]];
            }
            return;
        };

        function<bool(int)> augment_check = [&](int u) {
            first[u] = label[u] = 0;
            q.emplace(u);
            while (not q.empty()) {
                int x = q.front();
                q.pop();
                for (auto e : g[x]) {
                    int y = e.first;
                    if (mate[y] == 0 and y != u) {
                        mate[y] = x;
                        rematch(x, y);
                        return true;
                    } else if (label[y] >= 0) {
                        assignLabel(x, y, e.second);
                    } else if (label[mate[y]] < 0) {
                        label[mate[y]] = x;
                        first[mate[y]] = y;
                        q.emplace(mate[y]);
                    }
                }
            }
            return false;
        };

        for (int i = 1; i <= n; i++) {
            q = queue<int>();
            if (mate[i] != 0) continue;
            if (augment_check(i)) fill(label.begin(), label.end(), -1);
        }

        vector<pair<int, int>> ans;
        for (int i = 1; i <= n; i++) {
            if (i < mate[i]) ans.emplace_back(i - 1, mate[i] - 1);
        }
        return ans;
    }


snippet     fft
abbr        高速フーリエ変換

    typedef complex<double> Complex;

    vector<Complex> fft(vector<Complex> A, int n, int sgn = 1) {

        if (n == 1) return A;

        vector<Complex> f0, f1;
        for (int i = 0; i < (n / 2); i++) {
            f0.push_back(A[i * 2 + 0]);
            f1.push_back(A[i * 2 + 1]);
        }

        f0 = fft(f0, n / 2, sgn), f1 = fft(f1, n / 2, sgn);

        Complex zeta = Complex(cos(2.0 * M_PI / n), sin(2.0 * M_PI / n) * sgn);
        Complex pow_zeta = 1;

        for (int i = 0; i < n; i++) {
            A[i] = f0[i % (n / 2)] + pow_zeta * f1[i % (n / 2)];
            pow_zeta *= zeta;
        }
        return A;
    }

    vector<Complex> inv_fft(vector<Complex> A, int n) {
        auto ret = fft(A, n, -1);
        for (int i = 0; i < n; i++) { ret[i] /= n; }
        return ret;
    }

    vector<Complex> multiply(vector<Complex>& X, vector<Complex>& Y) {
        int n = 1;
        while (n < (X.size() + Y.size() + 1)) n *= 2;

        vector<Complex> ret;

        X.resize(n), Y.resize(n);
        X = fft(X, n, 1), Y = fft(Y, n, 1);

        for (int i = 0; i < n; i++) { ret.push_back(X[i] * Y[i]); }
        return inv_fft(ret, n);
    }

snippet     ext-gcd
abbr        拡張ユークリッドの互除法
    /* ax + by = gcd(a,b) の解 (x,y) を求め，
     * d = gcd(a,b) を返す．
     */
    int extgcd(int a, int b, int &x, int &y){
        int d = a;
        if(b != 0){
            d = extgcd(b, a % b, y, x);
            y -= (a / b) * x;
        }else{
            x = 1; y = 0;
        }
        return d;
    }
snippet     gcd
abbr        最大公約数
    int gcd(int x, int y) {
        int r;
        if(x < y) swap(x, y);

        while(y > 0){
            r = x % y;
            x = y;
            y = r;
        }
        return x;
    }
snippet     liner-congruence
abbr        線型連立合同式
    /* mod mでのaの逆元を求める.(gcd(a,m) = 1) */
    int modinv(int a, int m) {
        int x, y;
        extgcd(a, m, x, y);
        return (m + x % m) % m;
    }

    /* 連立線形方程式を解く.
            A[i] * x ≡ B[i] (mod M[i])
        なる方程式の解が
            x ≡ b (mod m)
        とかけるとき、(b, m)を返す.(存在しなければ(0, -1)) */
    pair<int, int> liner_congruence(vector<int> A, vector<int> B, vector<int> M) {
        int x = 0, m = 1;

        for (int i = 0; i < A.size(); ++i) {
            int a = A[i] * m, b = B[i] - A[i] * x, d = gcd(M[i], a);

            if (b % d != 0) // 解なし
                return make_pair(0, -1);

            int t = b / d * modinv(a / d, M[i] / d) % (M[i] / d);
            x = x + m * t;
            m *= M[i] / d;
        }

        return make_pair((x % m + m) % m, m);
    }

#segTree_rangeUpdate_rangeMin
# 区間更新、区間min、遅延評価
#segTree_rangeAdd_rangeSum
# 区間add、区間和
#segTree_rangeAdd_rangeMin
# 区間add、区間min、遅延評価
#segTree_pointUpdate
# 点更新、区間累積
#binaryIndexedTree
# 区間和

snippet     binaryIndexedTree
abbr        区間和
 const int MAX_N = 200000;
 
 template <typename T>
 class BIT{
 	private:
 		vector<T> bit; //[1, n]
 	public:
 		BIT(){
 			bit = vector<T>(MAX_N + 1, 0);
 		}
 		T sum(int i){
 			assert(1 <= i and i <= MAX_N and "invalid argument");
 			T s = 0;
 			while(i > 0){
 				s += bit[i];
 				i -= i & -i;
 			}
 			return s;
 		}
 
 		void add(int i, int x){
 			assert(1 <= i and i <= MAX_N and "invalid argument");
 			while(i <= MAX_N){
 				bit[i] += x;
 				//bit[i] = max(bit[i], x);
 				i += i & - i;
 			}
 		}
 };

snippet     segTree_pointUpdate
abbr        点更新、区間累積
 struct RMQ{
 	using T = int; //モノイドの台集合の型
 	T operator()(const T &a, const T &b) { return min(a,b); } //二項演算
 	static constexpr T identity() { return INT_MAX; } //単位元
 };
 
 struct RSQ{
 	using T = int; //モノイドの台集合の型
 	T operator()(const T &a, const T &b) { return a + b; } //二項演算
 	static constexpr T identity() { return 0; } //単位元
 };
 
 template<class Monoid>
 class SegTree{ //SegTree<Monoid> seg(n);
 	private:
 		using T = typename Monoid::T; //台集合の型をエイリアス
 		Monoid op; //演算の関数オブジェクト
 		const int n; //列の長さ
 		vector<T> t; //内部二分木
 
 		void prop_to(int i) { //ノードiに伝搬
 			t[i] = op(t[2 * i], t[2 * i + 1]);
 		}
 
 	public:
 		SegTree(int n) : n(n), t(4 * n, op.identity()) {} //単位元で埋める初期化
 		SegTree(const vector<T> &v) : n(v.size()), t(2 * n){ //配列を用いて初期化
 			copy(begin(v), end(v), begin(t) + n);
 			for(int i = n - 1; i > -0; i--) prop_to(i);
 		}
 
 		void set(int i, const T &x){
 			t[i += n] = x; 
 			while(i >>= 1) prop_to(i);
 		}
 
 		void add(int i, const T &x){
 			set(i, get(i) + x);
 		}
 
 		T get(int i) { return t[i + n]; } //列のi番目を返す
 
 		T query(int l, int r){ // 1-index
 			T accl = op.identity(), accr = op.identity();
 			for(l += n, r += n; l < r; l >>= 1, r >>= 1){
 				if(l & 1) accl = op(accl, t[l++]);
 				if(r & 1) accr = op(t[r-1], accr);
 			}
 			return op(accl, accr);
 		}
 };

snippet     segTree_rangeAdd_rangeMin
abbr        区間add、区間min、遅延評価
 struct RMQ{
 using T = long long;
 T operator()(const T &a, const T &b) { return min(a,b); };
 static constexpr T identity() { return 1LL << 60; }
 static constexpr T init() { return 0; }
 };
 
 template<class Monoid>
 class rangeAddQuery{
 private:
 	using T = typename Monoid::T;
 	Monoid op;
 	const int n;
 	vector<T> dat, lazy;
 	T query(int a, int b, int k, int l, int r){
 		eval(k);
 		if(b <= l || r <= a) return op.identity();
 		else if(a <= l && r <= b) return dat[k];
 		else{
 			T left = query(a, b, k * 2, l, (l + r) / 2);
 			T right = query(a, b, k * 2 + 1, (l + r) / 2, r);
 			propTo(k);
 			return op(left, right);
 		}
 	}
 	void add(int a, int b, int k, int l, int r, T x){
 		eval(k);
 		if(a <= l && r <= b){
 			lazy[k] += x;
 			eval(k);
 		}else if(l < b && a < r){
 			add(a, b, k * 2, l, (l + r) / 2, x);
 			add(a, b, k * 2 + 1, (l + r) / 2, r, x);
 			propTo(k);
 		}
 	}
 	inline void eval(int k){
 		if(lazy[k] == op.init()) return;
 		dat[k] += lazy[k];
 		if(k < n){
 			lazy[k * 2] += lazy[k];
 			lazy[k * 2 + 1] += lazy[k];
 		}
 		lazy[k] = op.init();
 	}
 	inline void propTo(int k){
 		dat[k] = op(dat[k * 2], dat[k * 2 + 1]);
 	}
 	int power(int n){
 		int res = 1;
 		while(n >= res) res*=2;
 		return res;
 	}
 public:
 	rangeAddQuery(int n) : n(power(n)), dat(4 * n, op.identity()), lazy(4 * n, op.init()) {}
 	rangeAddQuery(const vector<T> &v) : n(power(v.size())), dat(4 * n), lazy(4 * n, op.init()){
 		copy(begin(v), end(v), begin(dat) + n);
 		for(int i = n - 1; i > 0; i--) propTo(i);
 	}
 	T query(int a, int b){ return query(a,b,1,0,n); }
 	void add(int s, int t, T x){ add(s, t, 1, 0, n, x); }
 	T get(int a){ return query(a, a + 1); };
 	void out(){
 		rep(i,n * 2){ cout << dat[i + 1] << ' '; } cout << endl;
 	}
 };

snippet     segTree_rangeAdd_rangeSum
abbr        区間add、区間和
 struct RAQ{
 	using T = long long;
 	T operator()(const T &a, const T &b) { return a + b; };
 	static constexpr T identity() { return 0; }
 };
 
 template<class Monoid>
 class rangeAddQuery{
 	private:
 		using T = typename Monoid::T;
 		Monoid op;
 		const int n;
 		vector<T> dat, lazy;
 		T query(int a, int b, int k, int l, int r){
 			if(b <= l || r <= a) return op.identity();
 			else if(a <= l && r <= b) return dat[k] * (r - l) + lazy[k];
 			else{
 				T res = (min(b,r) - max(a,l)) * dat[k];
 				res += query(a, b, k * 2, l, (l + r) / 2);
 				res += query(a, b, k * 2 + 1, (l + r) / 2, r);
 				return res;
 			}
 		}
 		void add(int a, int b, int k, int l, int r, T x){
 			if(a <= l && r <= b){
 				dat[k] += x;
 			}else if(l < b && a < r){
 				lazy[k] += (min(b,r) - max(a,l)) * x;
 				add(a, b, k * 2, l, (l + r) / 2, x);
 				add(a, b, k * 2 + 1, (l + r) / 2, r, x);
 			
 			}
 		}
 		int power(int n){
 			int res = 1;
 			while(n >= res) res*=2;
 			return res;
 		}
 	public:
 		rangeAddQuery(int n) : n(power(n)), dat(4 * n, op.identity()), lazy(4 * n, op.identity()) {}
 		rangeAddQuery(const vector<T> &v) : n(power(v.size())), dat(4 * n), lazy(4 * n, op.identity()){
 			copy(begin(v), end(v), begin(dat) + n);
 			for(int i = n - 1; i > 0; i--) dat[i] = op(dat[2 * i], dat[2 * i + 1]);
 		}
 		T query(int a, int b){ return query(a,b,1,0,n); }
 		void add(int s, int t, T x){ add(s, t, 1, 0, n, x); }
 		T get(int a){ return query(a, a + 1); };
 		void out(){
 			rep(i,n * 2){ cout << dat[i + 1] << ' '; } cout << endl;
 		}
 };

snippet     segTree_rangeUpdate_rangeMin
abbr        区間更新、区間min、遅延評価
 struct RMQ{
 	using T = int;
 	T operator()(const T &a, const T &b) { return min(a,b); };
 	static constexpr T identity() { return INT_MAX; }
 };
 
 template<class Monoid>
 class rangeUpdateQuery{
 	private:
 		using T = typename Monoid::T;
 		Monoid op;
 		const int n;
 		vector<T> dat, lazy;
 		T query(int a, int b, int k, int l, int r){
 			evaluate(k);
 
 			if(b <= l || r <= a) return op.identity();
 			else if(a <= l && r <= b) return dat[k];
 			else{
 				int vl = query(a, b, k * 2, l, (l + r) / 2);
 				int vr = query(a, b, k * 2 + 1, (l + r) / 2, r);
 				return op(vl, vr);
 			}
 		}
 		inline void evaluate(int k){
 			if(lazy[k] == op.identity()) return;
 			dat[k] = lazy[k];
 			if(k < n){
 				lazy[2 * k] = lazy[k];
 				lazy[2 * k + 1] = lazy[k];
 			}
 			lazy[k] = op.identity();
 		}
 		inline void propTo(int k){
 			dat[k] = op(dat[2 * k], dat[2 * k + 1]);
 		}
 		void update(int a, int b, int k, int l, int r, T x){
 			evaluate(k);
 			if(r <= a || b <= l) return;
 			if(a <= l && r <= b){
 				lazy[k] = x;
 				evaluate(k);
 			}else if(l < b && a < r){
 				update(a, b, k * 2, l, (l + r) / 2, x);
 				update(a, b, k * 2 + 1, (l + r) / 2, r, x);
 				propTo(k);
 				evaluate(k);
 			}
 		}
 		int power(int n){
 			int res = 1;
 			while(n >= res) res*=2;
 			return res;
 		}
 	public:
 		rangeUpdateQuery(int n) : n(power(n)), dat(4 * n, op.identity()), lazy(4 * n, op.identity()) {}
 		rangeUpdateQuery(const vector<T> &v) : n(power(v.size())), dat(4 * n), lazy(4 * n, op.identity()){
 			copy(begin(v), end(v), begin(dat) + n);
 			for(int i = n - 1; i > 0; i--) propTo(i);
 		}
 		T query(int a, int b){ return query(a,b,1,0,n); }
 		void update(int s, int t, int x){ update(s, t, 1, 0, n, x); }
 		T get(int a){ return query(a, a + 1); };
 		void out(int n){
 			rep(i,n + 1){ cout << get(i) << ' '; }
 			cout << endl;
 			//rep(i,n){ range(j,i,n){ cout << i << ' ' << j + 1 << ' ' << query(i + 0,j + 1) << endl; } }
 		}
 		void out(){
 			rep(i,n * 2){ cout << i << " " << dat[i + 1] <<endl; }
 		}
 };
#partitionNumber
# 分割数
#derangement
# 完全順列（a[i] != iとなるような順列）の数を返す

snippet     derangement
abbr        完全順列（a[i] != iとなるような順列）の数を返す
 long long derangement(long long n /*長さ*/){
 	long long dp[1000000] = {0};
 	dp[2] = 1;
 	range(i,3,n + 1){
 		dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2]);
 		dp[i] %= M;
 	}
 	return dp[n];
 }

snippet     partitionNumber
abbr        分割数
 const long long M = 1000000007;
 // n 個を m 個に分割する
 vector<vector<long long>> partitionNumber(int n, int m){
 	vector<vector<long long>> dp(n + 1, vector<long long>(m, 0));
 	range(i,1,m + 1){
 		rep(j,n + 1){
 			if(j - i >= 0){
 				dp[i][j] = (dp[i - 1][j] + dp[i][j - i]) % M;
 			}else{
 				dp[i][j] = dp[i - 1][j];
 			}
 		}
 	}
 	return dp;
 }
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
#diameterOfTree
# 木の直径
#unionFind-WeightedVer
# 頂点重み付き
#unionFind
# 集合
#minCostFlow
# 最小費用流
#lca
# lowest common ancestor
#twoEdgeConnectedComponent
# 二重連結成分分解、無向辺の強連結成分分解
#articulationPoins
# 関節点と橋を全列挙
#dijkstra-list
# 隣接リストのダイクストラ
#kruskal
# クラスカル法による最小全域木構築
#dijkstra-matrix
# 隣接行列のダイクストラ
#stronglyConnectedComponents
# 強連結成分分解

snippet     stronglyConnectedComponents
abbr        強連結成分分解
 class StronglyConnectedComponents{
 	private:
 		vector<bool> used;
 		vector<int> vs; //帰りがけ順の並び
 		void dfs(int v){
 			used[v] = true;
 			rep(i,g[v].size()){
 				if(not used[g[v][i]]) dfs(g[v][i]);
 			}
 			vs.emplace_back(v);
 		}
 		void rdfs(int v, int k){
 			used[v] = true;
 			cmp[v] = k;
 			rep(i,rg[v].size()){
 				if(not used[rg[v][i]]) rdfs(rg[v][i], k);
 			}
 		}
 	public:
 		typedef vector<vector<int>> graph;
 		const int v; // 頂点数
 		int nv; // SCCした後の頂点数
 		graph g, rg; // グラフ、辺が逆になったグラフ
 		vector<int> cmp; //属する強連結成分のトポロジカル順序
 
 		StronglyConnectedComponents(int v) : used(v), v(v), g(v), rg(v), cmp(v) { }
 
 		void addEdge(int from, int to){
 			g[from].emplace_back(to);
 			rg[to].emplace_back(from);
 		}
 		int solve(){ // 強連結成分分解をしたあとのグラフの頂点数を返す
 			fill(all(used),0);
 			vs.clear();
 			rep(i,v){
 				if(not used[i]) dfs(i);
 			}
 			fill(all(used),0);
 			int k = 0;
 			for(int i = vs.size() - 1; i >= 0; i--){
 				if(not used[vs[i]]) rdfs(vs[i], k++);
 			}
 			return nv = k;
 		}
 		graph getCssGraph(vector<vector<int>>& node){
 			node = vector<vector<int>>(nv); // node[i]:=SCCによって頂点iに同一視された頂点
 			graph res(nv); // CSSしたあとのグラフ
 			rep(i,v){
 				node[cmp[i]].emplace_back(i);
 				for(auto to : g[i]){
 					if(cmp[i] == cmp[to]) continue;
 					res[cmp[i]].emplace_back(cmp[to]);
 				}
 			}
 			return res;
 		}
 		void out(){
 			rep(i,v){ cout << cmp[i] << ' '; } cout << endl;
 		}
 };

snippet     dijkstra-matrix
abbr        隣接行列のダイクストラ
 const int MAX_V = 1000;
 int dijkstra(int n, int s, int g[MAX_V][MAX_V]){
 	vector<int> dis(n,INF);
 	priority_queue<int, vector<pair<int, int>>, greater<pair<int, int>>> q;
 
 	q.emplace(0,s);
 	dis[s] = 0;
 
 	while(not q.empty()){
 		int pos;
 		int cost;
 		tie(cost, pos) = q.top(); q.pop();
 		if(pos == 1) break;
 
 		rep(to,n){
 			if(dis[to] > cost + g[pos][to]){
 				dis[to] = cost + g[pos][to];
 				q.emplace(dis[to], to);
 			}
 		}
 	}
 	return dis[1];
 }

snippet     kruskal
abbr        クラスカル法による最小全域木構築
 class UnionFind{
 	private:
 		vector<int> par, depth;
 	public:
 		UnionFind() {}
 		UnionFind(int n){
 			init(n);
 		}
 		void init(int n){
 			par = vector<int>(n);
 			depth = vector<int>(n);
 			rep(i,n){
 				par[i] = i;
 				depth[i] = 0;
 			}
 		}
 		int find(int x){
 			if(par[x] == x){
 				return x;
 			}else {
 				return par[x] = find(par[x]);
 			}
 		}
 		void unite(int x, int y){
 			x = find(x);
 			y = find(y);
 			if(x == y) return;
 
 			if(depth[x] < depth[y]){
 				par[x] = y;
 			}else{
 				par[y] = x;
 				if(depth[x] == depth[y]) depth[x]++;
 			}
 		}
 		bool same(int x, int y){
 			return find(x) == find(y);
 		}
 };
 
 struct Edge{
 	int u, v, cost, id;
 	Edge() {}
 	Edge(int u, int v, int cost) : u(u), v(v), cost(cost) {}
 
 	bool operator < ( const Edge &right ) const {
 		return cost < right.cost ? 1 : (cost == right.cost ? (u < right.u ? 1 : (u == right.u ? (v == right.v) : 0)) : 0);
 	}
 };
 
 class Kruskal{
 	private:
 		static const int MAX_E = 50005;
 		UnionFind unionFind;
 	public:
 		int v;
 		vector<Edge> edge;
 		vector<int> mst; //edge[ mst[i] ] := 最小全域木を構築するi番目の辺
 		Kruskal(vector<Edge>& e, int _v){
 			v = _v;
 			unionFind.init(v);
 			edge = e;
 			sort(all(edge));
 		}
 		void reset(int v){
 			unionFind.init(v);
 		}
 		long long solve(){
 			long long res = 0;
 			rep(i,edge.size()){
 				Edge c = edge[i];
 				if(not unionFind.same(c.u, c.v)){
 					unionFind.unite(c.u, c.v);
 					res += c.cost;
 					mst.emplace_back(i);
 				}
 			}
 			int b = 0;
 			rep(i,v){
 				if(sinked[i]) continue;
 				b = i;
 				break;
 			}
 			rep(i,v){
 				if(not unionFind.same(b,i)){
 					return INF;
 				}
 			}
 			return res;
 		}
 };

snippet     dijkstra-list
abbr        隣接リストのダイクストラ
 struct Edge{
 	int to;
 	int cost;
 	Edge(int to, int cost) : to(to), cost(cost) {}
 };
 
 struct Node{
 	int dis;
 	bool used;
 	Node() : dis(INF), used(false) { }
 	Node(int d, bool f) : dis(d), used(f) { }
 };
 
 typedef vector<vector<Edge>> graph;
 
 int dijkstra(graph g, int s, int n){
 	vector<Node> node(n);
 	priority_queue<int, vector<pair<int, int>>, greater<pair<int, int>>> q;
 
 	q.push(make_pair(0, s));
 	node[s] = Node{0, true};
 
 	while(not q.empty()){
 		int dis, pos;
 		tie(dis, pos) = q.top(); q.pop();
 		node[pos].used = true;
 
 		for(auto e : g[pos]){
 			if(node[e.to].used == true) continue;
 			if(node[e.to].dis > dis + e.cost){
 				node[e.to].dis = dis + e.cost;
 				q.emplace(node[e.to].dis, e.to);
 			}
 		}
 	}
 	return node[0].dis;
 }


snippet     articulationPoins
abbr        関節点と橋を全列挙
 class Node{
 	public:
 		int ord; //DFSの訪問の順番
 		int par; //DFS Treeにおける親
 		int low; //min(自分のord, 逆辺がある場合の親のord, すべての子のlow)
 		Node() : ord(-1) { }
 };
 
 class ArticulationPoints {
 	private:
 		int v, cnt;
 		vector<Node> node;
 		void dfs(int cur, int prev){
 			node[cur].ord = node[cur].low = cnt;
 			cnt++;
 
 			for(auto to : g[cur]){
 				if(to == prev) continue;
 				if(node[to].ord >= 0){
 					node[cur].low = min(node[cur].low, node[to].ord);
 				}else{
 					node[to].par = cur;
 					dfs(to, cur);
 					node[cur].low = min(node[cur].low, node[to].low);
 				}
 				if(node[cur].ord < node[to].low){
 					bridge.emplace_back(min(cur, to), max(cur, to));
 				}
 			}
 		}
 	public:
 		vector<vector<int>> g;
 		vector<pair<int, int>> bridge;
 		set<int> ap; // 関節店
 		ArticulationPoints(int n) : v(n), cnt(1), node(n), g(n) { }
 		void addEdge(int a, int b){
 			g[a].emplace_back(b);
 			g[b].emplace_back(a);
 		}
 		bool isBridge(int u, int v){
 			if(node[u].ord > node[v].ord) swap(u,v);
 			return node[u].ord < node[v].low;
 		}
 		void run(){
 			dfs(0, -1); // 0 = root
 
 			int np = 0;
 			range(i,1,v){
 				int p = node[i].par;
 				if(p == 0) np++;
 				else if(node[p].ord <= node[i].low) ap.emplace(p);
 			}
 			if(np > 1) ap.emplace(0);
 			sort(all(bridge));
 			//for(auto it:ap){ cout << it << endl; } //関節点の全列挙
 			//for(auto it:bridge){ cout << it.first << ' ' << it.second << endl; } //橋の全列挙
 		}
 };

snippet     twoEdgeConnectedComponent
abbr        二重連結成分分解、無向辺の強連結成分分解
 class TwoEdgeConnectedComponent : public ArticulationPoints {
 	private:
 		void dfs(int c, int pos){
 			ver[c].emplace_back(pos);
 			comp[pos] = c;
 			for (int to : g[pos]) {
 				if (comp[to] >= 0) continue;
 				if (isBridge(pos, to)) continue;
 				dfs(c, to);
 			}
 		}
 		void addComp(int pos){
 			if(comp[pos] >= 0) return;
 			ver.emplace_back();
 			dfs(ver.size() - 1, pos);
 		}
 	public:
 		vector<int> comp; // 成分分解前の頂点から分解後の頂点への写像？
 		vector<vector<int>> ver; // 分解後の頂点と、その頂点に含まれる分解前の頂点
 		TwoEdgeConnectedComponent(int n) : ArticulationPoints(n), comp(n,-1) {}
 		void solve(){
 			run();
 			for(auto p : bridge){
 				addComp(p.first);
 				addComp(p.second);
 			}
 			addComp(0);
 		}
 		vector<vector<int>> getTree(){
 			vector<vector<int>> res(ver.size());
 			for(auto p : bridge){
 				int u = comp[p.first], v = comp[p.second];
 				res[u].emplace_back(v);
 				res[v].emplace_back(u);
 			}
 			return res;
 		}
 };

snippet     lca
abbr        lowest common ancestor
 class LCA{
 	private:
 		vector<vector<int>> g, parent;
 		int n, root, log_n;
 		bool inited;
 		vector<int> depth;
 		void dfs(int v, int p, int d){
 			parent[0][v] = p;
 			depth[v] = d;
 			rep(i,g[v].size()){
 				if(g[v][i] != p) dfs(g[v][i], v, d + 1);
 			}
 		}
 		void init(){
 			inited = true;
 			dfs(root, -1, 0);
 			rep(k,log_n - 1){
 				rep(i,n){
 					if(parent[k][i] < 0) parent[k + 1][i] = -1;
 					else parent[k + 1][i] = parent[k][ parent[k][i] ];
 				}
 			}
 		}
 		void dfs(int pos){
 			for(auto to : g[pos]){
 				if(dis[to] != -1) continue;
 				dis[to] = dis[pos] + 1;
 				dfs(to);
 			}
 		}
 	public:
 		vector<int> dis;
 		LCA(int n, int root = 0) : g(n), parent(log2(n) + 1, vector<int>(n)), n(n), root(root), log_n(log2(n) + 1), inited(false), depth(n), dis(n, -1) { }
 		void addEdge(int u, int v){
 			g[u].emplace_back(v);
 			g[v].emplace_back(u);
 		}
 		void dfs(){
 			dis[0] = 0;
 			dfs(0);
 		}
 		int dist(int u, int v){
 			return dis[u] + dis[v] - 2 * dis[get(u,v)];
 		}
 		int get(int u, int v){
 			if(not inited) init();
 			if(depth[u] > depth[v]) swap(u, v);
 			rep(k,log_n){
 				if( (depth[v] - depth[u]) >> k & 1){
 					v = parent[k][v];
 				}
 			}
 			if(u == v) return u;
 			for(int k = log_n - 1; k >= 0; k--){
 				if(parent[k][u] != parent[k][v]){
 					u = parent[k][u];
 					v = parent[k][v];
 				}
 			}
 			return parent[0][u];
 		}
 };

snippet     minCostFlow
abbr        最小費用流
 const int INF = 1e9;
 
 class Edge{
 	public:
 		//行き先、容量、コスト、逆辺
 		int to, cap, cost, rev;
 		Edge(int to, int cap, int cost, int rev) : to(to), cap(cap), cost(cost), rev(rev){}
 };
 
 class MinCostFlow {
 	int v;
 	vector<vector<Edge>> G;
 	vector<int> h; //ポテンシャル
 	vector<int> dist; //最短距離
 	vector<int> prev_v, prev_e; //直前の頂点と辺
 
 	public:
 	MinCostFlow(int n = 500) : v(n), G(n), h(n), dist(n), prev_v(n), prev_e(n) { }
 	void addEdge(int from, int to, int cap, int cost){
 		//cout << from << ' ' << to << ' ' << cap << ' ' << cost << endl;
 		G[from].emplace_back(Edge(to, cap, cost, static_cast<int>(G[to].size())));
 		G[to].emplace_back(Edge(from, 0, -cost, static_cast<int>(G[from].size() - 1)));
 	}
 	int bellmanFord(int s, int t, int f){
 		int res = 0;
 		while(f > 0){
 			dist = vector<int>(v,INF);
 			dist[s] = 0;
 			bool update = true;
 			while(update){
 				update = false;
 				rep(i,v){
 					if(dist[i] == INF) continue;
 					rep(j,G[i].size()){
 						Edge &e = G[i][j];
 						if(e.cap > 0 and dist[e.to] > dist[i] + e.cost){
 							dist[e.to] = dist[i] + e.cost;
 							prev_v[e.to] = i;
 							prev_e[e.to] = j;
 							update = true;
 						}
 					}
 				}
 			}
 			if(dist[t] == INF){
 				return -1;
 			}
 
 			int d = f;
 			for(int u = t; u != s; u = prev_v[u]){
 				d = min(d, G[prev_v[u]][prev_e[u]].cap);
 			}
 			f -= d;
 			res += d * dist[t];
 			for(int u = t; u != s; u = prev_v[u]){
 				Edge &e = G[prev_v[u]][prev_e[u]];
 				e.cap -= d;
 				G[u][e.rev].cap += d;
 			}
 		}
 		return res;
 	}
 	int dijkstra(int s, int t, int f){
 		int res = 0;
 		h = vector<int>(v, 0);
 		while(f > 0){
 			priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
 			dist = vector<int>(v,INF);
 			dist[s] = 0;
 			q.push(make_pair(0, s));
 			while(not q.empty()){
 				pair<int, int> p = q.top(); q.pop();
 				int u = p.second;
 				if(dist[u] < p.first) continue;
 				rep(i,G[u].size()){
 					Edge &e = G[u][i];
 					if(e.cap > 0 && dist[e.to] > dist[u] + e.cost + h[u] - h[e.to]){
 						dist[e.to] = dist[u] + e.cost + h[u] - h[e.to];
 						prev_v[e.to] = u;
 						prev_e[e.to] = i;
 						q.push(make_pair(dist[e.to], e.to));
 					}
 				}
 			}
 			if(dist[t] == INF){
 				return -1;
 			}
 			rep(i,v) h[i] += dist[i];
 
 			int d = f;
 			for(int u = t; u != s; u = prev_v[u]){
 				d = min(d, G[prev_v[u]][prev_e[u]].cap);
 			}
 			f -= d;
 			res += d * h[t];
 			for(int u = t; u != s; u = prev_v[u]){
 				Edge &e = G[prev_v[u]][prev_e[u]];
 				e.cap -= d;
 				G[u][e.rev].cap += d;
 			}
 		}
 		return res;
 	}
 };

snippet     unionFind
abbr        集合
 class UnionFind{
 	private:
 		vector<int> par, depth;
 	public:
 		vector<int> cnt; // その集合の頂点数
 		UnionFind() {}
 		UnionFind(int n){
 			init(n);
 		}
 		void init(int n){
 			par = vector<int>(n);
 			depth = vector<int>(n, 0);
 			cnt = vector<int>(n, 1);
 			rep(i,n){
 				par[i] = i;
 			}
 		}
 		int find(int x){
 			if(par[x] == x){
 				return x;
 			}else {
 				return par[x] = find(par[x]);
 			}
 		}
 		void unite(int x, int y){
 			x = find(x);
 			y = find(y);
 			if(x == y) return;
 
 			if(depth[x] < depth[y]){
 				par[x] = y;
 				cnt[y] += cnt[x];
 			}else{
 				par[y] = x;
 				cnt[x] += cnt[y];
 				if(depth[x] == depth[y]) depth[x]++;
 			}
 		}
 		bool same(int x, int y){
 			return find(x) == find(y);
 		}
 };

snippet     unionFind-WeightedVer
abbr        頂点重み付き
 template<typename T>
 class UnionFind{
 	private:
 		vector<int> par, depth;
 		vector<T> sum; // 集合の重みの総和
 	public:
 		vector<int> cnt; // その集合の頂点数
 		WeightedUnionFind() {}
 		WeightedUnionFind(int n, vector<T>& c){
 			init(n, c);
 		}
 		void init(int n, vector<T>& c){
 			par = vector<int>(n);
 			depth = vector<int>(n, 0);
 			cnt = vector<int>(n, 1);
 			sum = c;
 			rep(i,n){
 				par[i] = i;
 			}
 		}
 		int find(int x){
 			if(par[x] == x){
 				return x;
 			}else {
 				return par[x] = find(par[x]);
 			}
 		}
 		void unite(int x, int y){
 			x = find(x);
 			y = find(y);
 			if(x == y) return;
 
 			if(depth[x] < depth[y]){
 				par[x] = y;
 				cnt[y] += cnt[x];
 				sum[y] += sum[x];
 			}else{
 				par[y] = x;
 				cnt[x] += cnt[y];
 				sum[x] += sum[y];
 				if(depth[x] == depth[y]) depth[x]++;
 			}
 		}
 		bool same(int x, int y){
 			return find(x) == find(y);
 		}
 		T weight(int x){
 			return sum[find(x)];
 		}
 };

snippet     diameterOfTree
abbr        木の直径
 const int INF = 1e8;
 
 class Tree{
 	int n;
 	void bfs(int s, vector<int>& dis, vector<int>& pre){
 		queue<int> q;
 		dis[s] = 0;
 		q.emplace(s);
 
 		while(not q.empty()){
 			int u = q.front(); q.pop();
 			show(u)
 			for(auto e : g[u]){
 				if(dis[e.first] == INF){
 					dis[e.first] = dis[u] + e.second;
 					pre[e.first] = u;
 					q.emplace(e.first);
 				}
 			}
 		}
 	}
 	public:
 	vector<vector<pair<int, int>>> g;
 	vector<int> path;
 	Tree(int n) : n(n), g(n) {}
 	void addEdge(int a, int b, int c = 1){
 		g[a].emplace_back(b, c);
 		g[b].emplace_back(a, c);
 	}
 	int diameter(){
 		vector<int> pre(n, -1), dis(n, INF);
 		bfs(0, dis, pre);
 
 		int maxi = 0, tgt = 0;
 		rep(i,n){
 			if(dis[i] == INF) continue;
 			if(maxi < dis[i]){
 				maxi  = dis[i];
 				tgt = i;
 			}
 		}
 
 		pre = vector<int>(n, -1);
 		dis = vector<int>(n, INF);
 		bfs(tgt, dis, pre);
 		maxi = 0; //tgtから最も遠い接点の距離maxi
 		rep(i,n){
 			if(dis[i] == INF) continue;
 			if(maxi < dis[i]){
 				maxi  = dis[i];
 				tgt = i;
 			}
 		}
 
 		int pos = tgt;
 		while(pre[pos] != -1){
 			path.emplace_back(pre[pos]);
 			pos = pre[pos];
 		}
 		return maxi;
 	}
 	pair<int, int> center(){
 		if(path.empty()) diameter();
 
 		assert(path.size() == 0 and "path is empty");
 		if(path.size() % 2 == 0){
 			return make_pair(path[ path.size() / 2 - 1], path[ path.size() / 2]);
 		}else{
 			return make_pair(path[ path.size() / 2], -1);
 		}
 	}
 };
#rightHand
# 右手法
#dijkstra-extend
# 拡張dijkstra
#grundy
# グランディー数を求める再帰関数
#eulerGraph
# オイラー路の構築
#AhoCorasick
# 文字列を扱う木構造
#bitSizeSet
# 0 ~ n - 1 に含まれる、サイズkの部分集合を列挙する
#bitSubSet
# 部分集合のビット列を降順に列挙する
#levenshteinDistance
# 編集距離
#matrixChainMultiplication
# 連鎖行列積
#maximumrectangule
# 最大長方形
#cumulativeSum2D
# 2次元累積和
#maximumSubarray
# 連続した部分列の和の最大値を求める
#treeDP
# 全方位木DPによる木の直径の演算
#closedLoop
# 閉路の検出
#JoinInterval
# 区間の結合
#repalceAll
# 文字列の置き換え
#eulerPhi
# オイラー関数
#simultaneousLinearEquations
# 連立一次方程式
#flow
# 最大流
#matrix
# 行列計算
#meet_in_the_middle
# bitによる全通列挙
#compress coordinate
# 座標圧縮
#gridUnion-find
# グリッドグラフのユニオン木
#dp_Partial_sum_with_number_restriction
# 個数制限付き部分和
#pascals_triangle
# n個を選ぶ組み合わせの中、k個を選ぶ組み合わせの割合。
#intervalState
# 区間の関係
#toDAG
# 2点間の最短経路になる辺を残したDAG
#ternarySearch
# 三分探索
#numOfnumber
# nまでの数字を書いたとき、1が出現する回数
#next_combination
# 組み合わせの全列挙
#eightQueensProblem
# 8クイーン問題
#numPrimeFactor
# 約数の個数を求める
#bitManip
# ビットの操作
#oneStrokePath
# 全ての頂点を通る一筆書きの総数
#levenshteinDistance
# 編集距離を求めるdp
#bfsOfGrid
# グリッドの幅優先探索
#dfsOfTree
# 木の深さ優先探索
#eratosthenes
# 10^6以下の素数を全列挙
#GreatestCommonDivisor
# 最小公倍数
#lcm
# うるう年判定
#maze
# 迷路をグリッドに拡張
#nextMonth
# 日付の計算
#parser
# 構文解析
#power
# 冪乗
#rotate
# 行列の回転
#toNum
# 文字列から数値への変換
#toStr
# 数値から文字列への変換
#warshallFloyd
# 全点最短経路
#bellmanFord
# 負の経路を含む最短経路、負の経路の検出
#prim
# 最小無向全域木
#LCS
# 最長共通部分文字列の長さ
#LIS
# 最長増加部分列
#divisor
# 約数の列挙
#primeVactor
# 素因数分解
#syakutori
# しゃくとり
#isUniqueStr
# 文字列の文字がユニークか判定
#areAnagram
# 文字列がアナグラムかを判定
#spilt
# 文字列を空白で区切る
#topologicalSort
# トポロジカルソート
#heightOfTree
# 木の高さを求める
#bipartiteMatching
# 2部マッチング



snippet     bfsOfGrid
abbr        グリッドの幅優先探索

 const int N;
 
 struct Point{ int x, y; };
 int dy[4] = {0,1,0,-1}, dx[4] = {1,0,-1,0};
 bool M[N][N];
 
 int bfs(int h, int w, Point p){
     int dis[N][N];
     queue<Point> q;
     rep(i,N) rep(j,N) dis[i][j] = INF;
 
     dis[p.y][p.x] = 0;
     q.push(p);
 
     Point u;
     while(not q.empty()){
         u = q.front(); q.pop();
         rep(i,4){
             Point next;
             next.x = u.x + dx[i];
             next.y = u.y + dy[i];
             if(next.x < 0 || next.x >= w || next.y < 0 || next.y >= h) continue;
             if(dis[next.y][next.x] == INF && M[next.y][next.x]){
                 dis[next.y][next.x] = dis[u.y][u.x] + 1;
                 q.push(next);
             }
         }
     }
     return /*返り値*/;
 }


snippet     dfsOfTree
abbr        木構造の深さ優先探索
options     head

 typedef struct{
     int parent, left, right;
 } Node;
 Node t[1002];
 
 void dfs(int u, int d){
     if(t[u].left != INF){
         dfs(m, t[u].left, d + 1);
     }
     if(t[u].right != INF){
         dfs(m, t[u].right, d);
     }
 }
 
 int brotherNum(int u){
     if(t[u].right == INF){
         return u;
     }else{
         return brotherNum(t[u].right);
     }
 }
 
 void inputData(int par){
     t[i + 1].parent = par;
     if(t[par].left == INF){
         t[par].left = i + 1;
     }else{
         t[brotherNum(t[par].left)].right = i + 1;
     }
 }
 
 void printGraph(){
     range(i,1,n + 1){ cout << t[i].parent << ' ' << t[i].left << ' ' << t[i].right << endl; }
 }

snippet     eratosthenes
abbr        10^6以下の素数を全列挙
options     head
    
 const int kN;
 void primeNumber(bool prime[kN]){
     rep(i,kN) prime[i] = 1;
     prime[0] = prime[1] = 0;
     rep(i,kN){
         if(prime[i]){
             for(int j = i + i; j < kN; j+=i){
                 prime[j] = 0;
             }
         }
     }
 }

snippet     lcm
abbr        最小公倍数
options     head

 int gcd(int x, int y) {
    int r;
    if(x < y) swap(x, y);

    while(y > 0){
        r = x % y;
        x = y;
        y = r;
    }
    return x;
 }
    
 int lcm( int m, int n ) {
    // 引数に０がある場合は０を返す
    if ( ( 0 == m ) || ( 0 == n ) ) return 0;
    return ((m / gcd(m, n)) * n); // lcm = m * n / gcd(m,n)
 }

snippet     isPrime
abbr        素数判定
options     head
    
 bool primeNumber(int n){
    if(n < 2) return 0;
    else{
        for(int i = 2; i * i <= n; i++){
            if(n % i == 0) return 0;
        }
        return 1;
    }
 }

snippet     leapYear
abbr        うるう年判定
options     head
    
 /*
 うるう年判定
 400で割り切れる年数から数えて、0-399年間でうるう年は97回
 */
 bool leapYear(int y){
     if(y % 400 == 0 || (y % 4 == 0 && y % 100 != 0 )) return true;↲
     else return false;
 }

snippet     maze
abbr        迷路を二次平面上に拡張する
options     head
    
 const int N = ;

 void printMaze(int w, int h, bool M[N][N]){
    rep(i,h + h - 1){
        rep(j,w + w + 1){
            cout << M[i][j];
        }
        cout << endl;
    }
 }

 void extensionOfMaze(int w, int h, bool M[N][N]){
    rep(i,N) rep(j,N) M[i][j] = 1;
    rep(i,h + h - 1){
        if(i % 2){ //横線
            for(int j = 0; j <= w + w; j++){
                if(j == 0 || j == w + w) M[i + 1][j + 1] = 1;//壁
                else if(j % 2 == 0) M[i + 1][j + 1] = 1;
                else cin >> M[i + 1][j + 1];
            }
        }else{ //縦線
            for(int j = 0; j <= w + w; j++){
                if(j == 0 || j == w + w) M[i + 1][j + 1] = 1;//壁
                else if(j % 2 == 1) M[i + 1][j + 1] = 0;
                else cin >> M[i + 1][j + 1];
            }
        }
    }
 }

snippet     nextMonth
abbr        日付の計算
options     head

 bool isLeapYear(int y){
     if(y % 400 == 0 || (y % 4 == 0 && y % 100 != 0 )) return true;↲
     else return false;
 }
    
 void nextMonth(int &y, int &m, int &d){
    bool leapYear = isLeapYear(y);
    if((m == 2 && d == 30 && leapYear) ||
       (m == 2 && d == 29 && !leapYear) ||
       ((m == 4 || m == 6 || m == 9 || m == 11) && d == 31) ||
       ((m == 1 || m == 3 || m == 5 || m == 7 || m == 8 || m == 10 || m == 12) && d == 32)){
        m++;
        d = 1;
    }
    if(m == 13){
        y++;
        m = 1;
    }
 }

snippet     parser
abbr        構文解析
options     head
    
 typedef string::const_iterator State;
 int number(State&);
 int factor(State&);
 int term(State&);
 int expression(State&);
 
 // 数字の列をパースして、その数を返す。
 int number(State &begin) {
     int ret = 0;
 
     while (isdigit(*begin)) {
         ret *= 10;
         ret += *begin - '0';
         begin++;
     }
 
     return ret;
 }
 
 // 括弧か数をパースして、その評価結果を返す。
 int factor(State &begin) {
     if (*begin == '(') {
         begin++; // '('を飛ばす。
         int ret = expression(begin);
         begin++; // ')'を飛ばす。
         return ret;
     } else {
         return number(begin);
     }
     return 0;
 }
 
 // 乗算除算の式をパースして、その評価結果を返す。
 int term(State &begin) {
     int ret = factor(begin);
 
     for (;;) {
         if (*begin == '*') {
             begin++;
             ret *= factor(begin);
         } else if (*begin == '/') {
             begin++;
             ret /= factor(begin);
         } else {
             break;
         }
     }
 
     return ret;
 }
 
 // 四則演算の式をパースして、その評価結果を返す。
 int expression(State &begin) {
     int ret = term(begin);
 
     for (;;) {
         if (*begin == '+') {
             begin++;
             ret += term(begin);
         } else if (*begin == '-') {
             begin++;
             ret -= term(begin);
         } else {
             break;
         }
     }
 
     return ret;
 }

 //beginがexpectedを指していたらbeginを一つ進める。
 void consume(State &begin, char expected) {
     if (*begin == expected) {
         begin++;
     } else {
         cerr << "Expected '" << expected << "' but got '" << *begin << "'" << endl;
         cerr << "Rest string is '";
         while (*begin) {
             cerr << *begin++;
         }
         cerr << "'" << endl;
         //throw ParseError();
     }
 }

snippet     power
abbr        冪乗
options     head

 //x^n mod M
 typedef unsigned long long ull;
 const ull M = 1000000007;
 ull power(ull x, ull n){
     ull res = 1;
     if(n > 0){
         res = power(x, n / 2);
         if(n % 2 == 0) res = (res * res) % M;
         else res = (((res * res) % M) * x ) % M;
     }
     return res;
 }

snippet     rotate
abbr        行列の回転
options     head
 //P is structure of coodinate.
 void rotationMatrix(P &p, double angle){
     double x, y;
         x = p.x * cos(angle) - p.y * sin(angle);
         y = p.x * sin(angle) + p.y * cos(angle);
         p.x = x;
         p.y = y;
 }

snippet     toNum
abbr        文字列から数値への変換
options     head
    
 //文字列から数値への変換
 int toNum(string str){
    int num = 0;
    rep(i,str.size()){
        num *= 10;
        num += str[i] - '0';
    }
    return num;
 }

snippet     toStr
abbr        数値から文字列への変換
options     head
    
 string toStr(int n){
    string str;
    int len = static_cast<int>(log10(n));
    int K = 1;
    rep(i,len) K*=10;
    rep(i,len + 1){
        if(n / K == 0) str+= '0';
        else str+= ('0' + n / K);
        n%=K; K/=10;
    }
    return str;
 }

snippet     warshallFloyd
abbr        全点最短経路
options     head

 const int MAX_V = ;

 void init(int m[MAX_V][MAX_V]){
     rep(i,MAX_V) rep(j,MAX_V) m[i][j] = INF;
     rep(i,MAX_V) m[i][i] = 0;
 }

 void warshallFloyd(int m[MAX_V][MAX_V], int n){
     rep(k,n){
         rep(i,n){
             rep(j,n){
                 m[i][j] = min(m[i][j], m[i][k] + m[k][j]);
             }
         }
     }
 }


snippet     bellmanFord
abbr        負の経路を含む最短経路、負の経路の検出
options     head

 class Edge{
     public:
         int to, cost;
         Edge(int to, int cost) : to(to) ,cost(cost) {}
 };
 
 typedef vector<vector<Edge>> AdjList;
 vector<int> dis;
 
 bool bellmanFord(AdjList g, int n, int s) { // nは頂点数、sは開始頂点
     dis = vector<int>(n, INF);
     dis[s] = 0; // 開始点の距離は0
     rep(i,n){
         rep(v,n){
             rep(k,g[v].size()){
                 Edge e = g[v][k];
                 if (dis[v] != INF && dis[e.to] > dis[v] + e.cost) {
                     dis[e.to] = dis[v] + e.cost;
                     if (i == n - 1) return true; // n回目にも更新があるなら負の閉路が存在
                 }
             }
         }
     }
     return false;
 }
 
snippet     prim
abbr        最小全域木
options     head

 class Edge {
     public:
     int to, cost;
     Edge(int to, int cost) : to(to), cost(cost) { }
 };
 
 typedef vector<Edge> Edges;
 typedef vector<Edges> Graph;
 
 int prim(const Graph &g, int s = 0) {
     int n = g.size();
     int total = 0;
 
     vector<bool> visited(n);
     //priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
     priority_queue<pair<int, int> > q;
 
     q.push(make_pair(0,s));
     while (not q.empty()) {
         pair<int, int> u = q.top(); q.pop();
         if (visited[u.second]) continue;
         total += u.first;
         visited[u.second] = true;
         for(auto it : g[u.second]) {
             if (not visited[it.to]) q.push(make_pair(it.cost, it.to));
         }
     }
     return total;
 }

snippet     LCS
abbr        最長共通部分文字列の長さ
options     head

 const int MAX_N = 1000;
 
 int solve(string s, string t){
     int dp[MAX_N + 1][MAX_N + 1] = {{0}};

     rep(i,s.size()){
         rep(j,t.size()){
             if(s[i] == t[j]){
                 dp[i + 1][j + 1] = dp[i][j] + 1;
             }else{ //連続した部分文字列の場合はここを消し、dpのmaxを返す
                 dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j]);
             }
         }
     }
     return dp[n][m];
 }

snippet     LIS
abbr        最長増加部分列
options     head
 const int MAX_N = 1000;
 int dp[MAX_N];

 int LIS(int n, int a[MAX_N]){
     fill(dp, dp + n, INF);
     rep(i,n){
         *lower_bound(dp, dp + n, a[i]) = a[i];
     }
     return lower_bound(dp, dp + n, INF) - dp;
 }

snippet     divisor
abbr        約数の列挙
options     head
 vector<int> divisor(int n){
     vector<int> res;
     for(int i = 1; i * i <= n; i++){
         if(n % i == 0){
             res.emplace_back(i);
             if(i != n / i) res.emplace_back(n / i);
         }
     }
     return res;
 }

snippet     primeFactor
abbr        素因数分解
options     head
 map<int, int> primeFactor(int n){
     map<int, int> res;
     for(int i = 2; i * i <= n; i++){
         while(n % i == 0){
             ++res[i];
             n /= i;
         }
     }
     if(n != 1) res[n] = 1;
     return res;
 }

snippet     syakutori
abbr        しゃくとり
options     head
 //S:sumの条件 n:個数
 int solve(){
     int res = n + 1;
     int s = 0, t = 0, sum = 0;
     while(true){
         while(t < n && sum < S){
             sum += a[t++];
         }
         if(sum < S) break;
         res = min(res, t - s);
         sum -= a[s++];
     }
     if(res > n){
         res = 0;
     }
     return res;
 }

snippet     isUniqueStr
abbr        文字列の文字がユニークか判定
options     head
    
 bool isUniqueStr(string s){
     unsigned int char_set[128] = {0};
     rep(i,s.size()) char_set[s[i]]++;
     rep(i,128) if(char_set[i] >= 2) return 0;
     return 1;
 }

snippet     areAnagram
abbr        文字列がアナグラムかを判定
options     head

 bool areAnagram(string s, string t){
     if(s.size() != t.size()) return 0;
 
     unsigned int char_set[128] = {0};
     rep(i,s.size()){
         char_set[s[i]]++;
         char_set[t[i]]--;
     }
     rep(i,128) if(char_set[i] != 0) return 0;
     return 1;
 }

snippet     split
abbr        文字列を空白で区切る
options     head

 vector<string> split(string in, char sp = ' '){
     vector<string> ret;
     stringstream ss(in);
     string s;
     while(getline(ss, s, sp)){
         ret.emplace_back(s);
     }
     return ret;
 }


snippet     topologicalSort
abbr        トポロジカルソート
options     head

 const int MAX_V = 10000;
 
 vector<int> g[MAX_V]; //グラフ
 vector<int> tp; //トポロジカルソートの結果
 
 void bfs(int s, int indeg[MAX_V], bool used[MAX_V]){
     queue<int> q;
 
     q.push(s);
     used[s] = true;
 
     while(not q.empty()){
         int u = q.front(); q.pop();
         tp.emplace_back(u);
         rep(i,g[u].size()){
             int v = g[u][i];
             indeg[v]--;
             if(indeg[v] == 0 && not used[v]){
                 used[v] = true;
                 q.push(v);
             }
         }
     }
 }
 
 //グラフに閉路がある場合、0を返す
 bool topologicalSort(int v){
     int indeg[MAX_V]; //入次数
     bool used[MAX_V];
     memset(indeg, 0, sizeof(indeg));
     memset(used, 0, sizeof(used));
 
     rep(i,v) rep(j,g[i].size()) indeg[ g[i][j] ]++;
     rep(i,v) if(indeg[i] == 0 && not used[i]) bfs(i, indeg, used);
 
     for(auto it:tp) cout << it << endl;
 
     if(tp.size() == v) return true;
     else return false;
 }

snippet     heightOfTree
abbr        木の高さを求める
options     head
    
 const int MAX_V = 10000;
 
 class Edge{
     public:
         int dst, weight;
         Edge(){}
         Edge(int dst, int weight): dst(dst), weight(weight)  {}
 };
 
 typedef vector<vector<Edge>> Graph;
 
 Graph g(MAX_V);
 
 int visit(Graph &t, int i, int j) {
     if(t[i][j].weight >= 0) return t[i][j].weight;
     t[i][j].weight = g[i][j].weight;
     int u = t[i][j].dst;
     rep(k,t[u].size()) {
         if(t[u][k].dst == i) continue;
         t[i][j].weight = max(t[i][j].weight, visit(t,u,k) + g[i][j].weight);
     }
     return t[i][j].weight;
 }
 vector<int> height(int n) {
     Graph t = g;
     rep(i,n) rep(j,t[i].size()) t[i][j].weight = -1;
     rep(i,n) rep(j,t[i].size()) if(t[i][j].weight < 0) t[i][j].weight = visit(t, i, j);
 
     vector<int> ht(n); // gather results
     rep(i,n) rep(j,t[i].size()) ht[i] = max(ht[i], t[i][j].weight);
     return ht;
 }



snippet     bipartiteMatching
abbr        2部マッチング
options     head

 const int MAX_V = 210; //MAX_X + MAX_Y
 const int MAX_X = 105;
 const int MAX_Y = 105;
 
 class Edge{
     public:
         int to, cap, rev;
 };
 
 typedef vector<vector<Edge>> AdjList;
 AdjList G(MAX_V);
 bool used[MAX_V];
 
 void addEdge(int from, int to, int cap){
     G[from].emplace_back(Edge{to, cap, static_cast<int>(G[to].size())});
     G[to].emplace_back(Edge{from, 0, static_cast<int>(G[from].size() - 1)});
 }
 
 int dfs(int v, int t, int f){
     if(v == t) return f;
     used[v] = true;
     rep(i,G[v].size()){
         Edge &e = G[v][i];
         if(not used[e.to] && e.cap > 0){
             int d = dfs(e.to, t, min(f, e.cap));
             if(d > 0){
                 e.cap -= d;
                 G[e.to][e.rev].cap += d;
                 return d;
             }
         }
     }
     return 0;
 }
 
 int maxFlow(int s, int t){
     int flow = 0;
     while(true){
         memset(used, 0, sizeof(used));
         int f = dfs(s, t, INF);
         if(f == 0) return flow;
         flow += f;
     }
 }
 
 int bipartiteMatching(int x, int y, bool edge[MAX_X][MAX_Y]){
     int s = x + y, t = s + 1; //set x : 0 ~ x-1, set y : x ~ x+y-1
 
     rep(i,x) addEdge(s, i, 1); //sと集合xを結ぶ
     rep(i,y) addEdge(x + i, t, 1); //集合yとtを結ぶ
 
     rep(i,x) rep(j,y) if(edge[i][j]) addEdge(i, x + j, 1); //集合xと集合yを結ぶ
 
     return maxFlow(s, t);
 }

snippet     levenshteinDistance
abbr        編集距離を求めるdp
 const int MAX_N = 1005;
 int dp[MAX_N][MAX_N];
 
 int minimum(int a, int b, int c){
     return min(min(a,b),c);
 }
 
 int levenshteinDistance(string a, string b){
     rep(i,a.size() + 1) dp[i][0] = i;
     rep(i,b.size() + 1) dp[0][i] = i;
 
     range(i,1,a.size() + 1){
         range(j,1,b.size() + 1){
             int cost = a[i - 1] == b[j - 1] ? 0 : 1;
             dp[i][j] = minimum(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
         }
     }
     return dp[a.size()][b.size()];
 }

snippet     oneStrokePath
abbr        全ての頂点を通る一筆書きの総数
 const int MAX_V = 10;
 
 int n;
 int ans = 0;
 bool M[MAX_V][MAX_V] = {0};
 
 void dfs(int c, int u, vector<bool> used){
     if(u == n - 1){
         ans++;
         return;
     }
     rep(i,n){
         if(M[c][i] == 1 && used[i] == false){
             used[c] = true;
             dfs(i,u + 1,used);
         }
     }
 }

snippet     bitManip
abbr        ビットの操作
 //i番目のビットを返す
 bool getBit(int num, int i){
     return ((num & (1 << i)) != 0);
 }
 
 //i番目を1にする
 int setBit(int num, int i){
     return num | (1 << i);
 }
 
 //i番目を0にする
 int clearBit(int num, int i){
     int mask = ~(1 << i);
     return num & mask;
 }
 
 //i番目をvで置き換える
 int updateBit(int num, int i, int v){
     int mask = ~(1 << i);
     return (num & mask) | (v << i);
 }

snippet     numPrimeFactor
abbr        約数の個数を求める
 const long long M = 1000000007;
 map<int, int> res;
 
 void primeFactor(int n){
     for(int i = 2; i * i <= n; i++){
         while(n % i == 0){
             ++res[i];
             n /= i;
         }
     }
     if(n != 1) res[n] += 1;
 }
 
 long long numPrimeFactor(){
     long long ans = 1;
     for(auto it:res){
         ans = ans * (it.second + 1);
         ans %= M;
     }
     return ans;
 }

snippet     eightQueensProblem
abbr        8クイーン問題
 const int N = 8;
 
 //row[i] = j : (i,j)にクイーン
 int row[N], col[N], dpos[2 * N - 1], dneg[2 * N - 1];
 char x[8][8];
 bool f = false;
 
 void init(){
     rep(i,N){
         row[i] = 0;
         col[i] = 0;
     }
     rep(i,2 * N - 1){
         dpos[i] = 0;
         dneg[i] = 0;
     }
 }
 
 void printBoard(){
     rep(i,N){
         rep(j,N){
             if(x[i][j] == 'Q'){
                 if(row[i] != j) return;
             }
         }
     }
     rep(i,N){
         rep(j,N){
             cout << ( (row[i] == j) ? "Q" : ".");
         }
         cout << endl;
     }
     f = true;
 }
 
 void recursive(int i){
     if(i == N){
         printBoard();
         return;
     }
 
     rep(j,N){
         if(col[j] || dpos[i + j] || dneg[i - j + N - 1]) continue;
         row[i] = j;
         col[j] = dpos[i + j] = dneg[i - j + N - 1] = 1;
         recursive(i + 1);
         row[i] = col[j] = dpos[i + j] = dneg[i - j + N - 1] = 0;
         if(f) return;
     }
 }

snippet     next_combination
abbr        組み合わせの全列挙
 template < class BidirectionalIterator >
 bool next_combination ( BidirectionalIterator first1 ,
         BidirectionalIterator last1 ,
         BidirectionalIterator first2 ,
         BidirectionalIterator last2 ){
     if (( first1 == last1 ) || ( first2 == last2 )) {
         return false ;
     }
     BidirectionalIterator m1 = last1 ;
     BidirectionalIterator m2 = last2 ; --m2;
     while (--m1 != first1 && !(* m1 < *m2 )){
     }
     bool result = (m1 == first1 ) && !(* first1 < *m2 );
     if (! result ) {
         while ( first2 != m2 && !(* m1 < * first2 )) {
             ++ first2 ;
         }
         first1 = m1;
         std :: iter_swap (first1 , first2 );
         ++ first1 ;
         ++ first2 ;
     }
     if (( first1 != last1 ) && ( first2 != last2 )) {
         m1 = last1 ; m2 = first2 ;
         while (( m1 != first1 ) && (m2 != last2 )) {
             std :: iter_swap (--m1 , m2 );
             ++ m2;
         }
         std :: reverse (first1 , m1 );
         std :: reverse (first1 , last1 );
         std :: reverse (m2 , last2 );
         std :: reverse (first2 , last2 );
     }
     return ! result ;
 }
 
 template < class BidirectionalIterator > bool next_combination ( BidirectionalIterator first , BidirectionalIterator middle , BidirectionalIterator last )
 {
     return next_combination (first , middle , middle , last );
 }
 
 //要素vからr個取り出す組み合わせ
 void func(vector<int> v, int r){
     do{
     }while(next_combination(v.begin(), v.begin() + r, v.end()));
 }

snippet     numOfnumber
abbr        nまでの数字を書いたとき、1が出現する回数
 //nに含まれる1の数
 long long numOfnumber(long long n){
     long long k = 10;
     long long ans = 0;
     rep(i,9){
         ans += n / k * (k / 10LL);
         ans += min(max(0LL, n % k - (k / 10LL - 1)), k / 10LL);
         k*=10LL;
     }
     return ans;
 }

snippet     ternarySearch
abbr        三分探索
 double C(double x){
 }
 
 double ternarySearch(double p){
     double right = INF, left = 0;
     rep(i,200){
         double llr = (left * 2 + right) / 3;
         double rll = (left + right * 2) / 3;
         if(C(llr) > C(rll)){
             left = llr;
         }else{
             right = rll;
         }
     }
     return left;
 }


snippet     toDAG
abbr        2点間の最短経路になる辺を残したDAG
 //出発地、出発地から全ての点へ対する最短経路、返り値、辺
 void toDAG(int s, int g[MAX_V][MAX_V], int dag[MAX_V][MAX_V], vector<pair<int, int>> v){
     rep(i,v.size()){
         if(g[s][v[i].first] + 1 == g[s][v[i].second]){
             dag[v[i].first][v[i].second] = 1;
         }
     }
 }

snippet     intervalState
abbr        区間の関係
 //区間A[a,b]と区間B[c,d]の関係
 int intervalState(int a, int b, int c, int d){
     if(a < c && b < c) return 0;            //A < B
     else if(a > d && b > d) return 1;       //A > B
     else if(a <= c && d <= b) return 2;     //A -> B
     else if(c < a && b < d) return 3;       //B -> A
     else if(a <= c && b < d) return 4;      //A <= B
     else if(c < a && d <= b) return 5;      //A >= B
     return -1;
 }

snippet     pascals_triangle
abbr        n個を選ぶ組み合わせの中、k個を選ぶ組み合わせの割合。
 //n個を選ぶ組み合わせの中、k個を選ぶ組み合わせの割合。
 void Pascals(double m[N][N]){
     m[0][0] = 1;
     range(i,1,1011){
         rep(j,i + 1){
             if(j == 0) m[i][j] = m[i - 1][j] / 2;
             else if(j == i) m[i][j] = m[i - 1][j - 1] / 2;
             else m[i][j] = (m[i - 1][j] + m[i - 1][j - 1]) / 2;
         }
     }
 }

snippet     dp_Partial_sum_with_number_restriction
abbr        個数制限付き部分和

 const int MAX_N = 105;
 const int MAX_K = 100005;

 void solve(){
     int n, k;
 
     scanf("%d%d", &n, &k);
     int a[MAX_N], m[MAX_N];
     rep(i,n) scanf("%d", &a[i]);
     rep(i,n) scanf("%d", &m[i]);
 
     int dp[MAX_K];
     memset(dp, -1, sizeof(dp));
     dp[0] = 0;
     rep(i,n){
         rep(j,k + 1){
             if(dp[j] >= 0){
                 dp[j] = m[i];
             }else if(j < a[i] || dp[j - a[i]] <= 0){
                 dp[j] = -1;
             }else{
                 dp[j] = dp[j - a[i]] - 1;
             }
         }
     }
 
     int sum = 0;
     range(i,1,k + 1){
         if(dp[i] >= 0) sum++;
     }
     cout << sum << endl;
 }

snippet     gridUnion-find
abbr        グリッドグラフのユニオン木
 const int gmax_n =1005 ;
 const int dy[3] = { 0, 1, 1};
 const int dx[3] = { 1, 0, 1};
 
 pair<int, int> par[gmax_n][gmax_n]; //親
 int depth[gmax_n][gmax_n];//木の深さ
 int cc[gmax_n][gmax_n]; //連結成分
 
 void init(int h, int w){
     rep(i,h){
         rep(j,w){
             par[i][j] = make_pair(i,j);
             depth[i][j] = 0;
             cc[i][j] = 1;
         }
     }
 }
 
 pair<int, int> find(pair<int, int> x){
     if(par[x.first][x.second] == x){
         return x;
     }else {
         return par[x.first][x.second] = find(par[x.first][x.second]);
     }
 }
 
 void unite(pair<int, int> x, pair<int, int> y){
     x = find(x);
     y = find(y);
     if(x == y) return;
 
     if(depth[x.first][x.second] < depth[y.first][y.second]){
         par[x.first][x.second] = y;
         cc[y.first][y.second] += cc[x.first][x.second];
     }else{
         par[y.first][y.second] = x;
         cc[x.first][x.second] += cc[y.first][y.second];
         if(depth[x.first][x.second] == depth[y.first][y.second]) depth[x.first][x.second]++;
     }
 }
 
 bool same(pair<int, int> x, pair<int, int> y){
     return find(x) == find(y);
 }
 
 void uniteAll(int h, int w, char m[gmax_n][gmax_n]){
     rep(i,h){
         rep(j,w){
             if(m[i][j] == 'o'){
                 rep(k,3){
                     int ny = i + dy[k];
                     int nx = j + dx[k];
                     if(ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                     if(m[ny][nx] == 'o') unite(make_pair(i,j), make_pair(ny,nx));
                 }
             }
             if(i < h && j < w && m[i + 1][j] == 'o' && m[i][j + 1] == 'o'){
                 unite(make_pair(i + 1, j), make_pair(i, j + 1));
             }
         }
     }
 }
 
 void check(int n, int ans[3]){
     int i = 0;
     while(true){
         i++;
         assert(i <= 310);
         if(n % (i * i) != 0) continue;
 
         if(n / (i * i) == 12){ ans[0]++; break; }
         else if(n / (i * i) == 16){ ans[1]++; break; }
         else if(n / (i * i) == 11){ ans[2]++; break; }
     }
 }
 
 void print(int h, int w, char m[gmax_n][gmax_n]){
     rep(i,h){ rep(j,w){ cout << cc[i][j] << ' '; } cout << endl; } cout << endl;
     rep(i,h){ rep(j,w){
         if(make_pair(i,j) == par[i][j]) cout << '.';
         else cout << '#';
     } cout << endl; } cout << endl;
 }


snippet     meet_in_the_middle
abbr        bitによる全通列挙
 //要素wをnとmに分け、それぞれで全列挙する
 vector<long long> a, b;
 rep(i,(1 << n)){
     long long sum = 0;
     rep(j,n){
         if(getBit(i,j)) sum += w[j];
     }
     a.emplace_back(sum);
 }
 rep(i,(1 << m)){
     long long sum = 0;
     rep(j,m){
         if(getBit(i,j)) sum += w[n + j];
     }
     b.emplace_back(sum);
 }

snippet     matrix
abbr        行列計算
 const int M = 10000;
 typedef vector<vector<int>> mat;
 
 mat mul(mat &a, mat &b){
     mat c(a.size(), vector<int>(b[0].size()));
     rep(i,a.size()){
         rep(k,b.size()){
             rep(j,b[0].size()){
                 c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % M;
             }
         }
     }
     return c;
 }
 
 mat pow(mat a, int n){
     mat b(a.size(), vector<int>(a.size()));
     rep(i,a.size()){
         b[i][i] = 1;
     }
     while(n > 0){
         if(n & 1) b = mul(b,a);
         a = mul(a, a);
         n >>= 1;
     }
     return b;
 }
 
 int solve(int n){
     mat a(2, vector<int>(2));
     a[0][0] = 1; a[0][1] = 1;//フィボナッチ数列の漸化式の行列
     a[1][0] = 1; a[1][1] = 0;
     a = pow(a,n); //行列Aのn乗。
     return a[1][0];
 }

snippet     flow
abbr        最大流
 const int MAX_V = 10005;
 
 class Edge{
     public:
         int to, cap, rev;
         Edge(int to, int cap, int rev) : to(to), cap(cap), rev(rev) {}
 };
 
 class Flow{
     private:
         vector<Edge> G[MAX_V];
         bool used[MAX_V];
         int level[MAX_V]; //sからの距離
         int iter[MAX_V]; //どこまで調べ終わったか
         int dfs(int v, int t, int f){
             if(v == t) return f;
             used[v] = true;
             rep(i,G[v].size()){
                 Edge &e = G[v][i];
                 if(not used[e.to] && e.cap > 0){
                     int d = dfs(e.to, t, min(f, e.cap));
                     if(d > 0){
                         e.cap -= d;
                         G[e.to][e.rev].cap += d;
                         return d;
                     }
                 }
             }
             return 0;
         }
         int dfs_(int v, int t, int f){
             if(v == t) return f;
             for(int &i = iter[v]; i < G[v].size(); i++){
                 Edge &e = G[v][i];
                 if(e.cap > 0 && level[v] < level[e.to]){
                     int d = dfs_(e.to, t, min(f, e.cap));
                     if(d > 0){
                         e.cap -= d;
                         G[e.to][e.rev].cap += d;
                         return d;
                     }
                 }
             }
             return 0;
         }
         void bfs(int s){
             memset(level, -1, sizeof(level));
             queue<int> que;
             level[s] = 0;
             que.push(s);
             while(not que.empty()){
                 int v = que.front(); que.pop();
                 rep(i,G[v].size()){
                     Edge &e = G[v][i];
                     if(e.cap > 0 && level[e.to] < 0){
                         level[e.to] = level[v] + 1;
                         que.push(e.to);
                     }
                 }
             }
         }
     public:
         void addEdge(int from, int to, int cap){
             G[from].push_back(Edge(to, cap, static_cast<int>(G[to].size())));
             G[to].push_back(Edge(from, 0, static_cast<int>(G[from].size() - 1)));
         }
         int fordFulkerson(int s, int t){
             int flow = 0;
             while(true){
                 memset(used, 0, sizeof(used));
                 int f = dfs(s, t, INF);
                 if(f == 0) return flow;
                 flow += f;
             }
         }
         int dinic(int s, int t){
             int flow = 0;
             while(true){
                 bfs(s);
                 if(level[t] < 0) return flow;
                 memset(iter, 0, sizeof(iter));
                 int f;
                 while( (f = dfs_(s, t, INF)) > 0){
                     flow += f;
                 }
             }
         }
 };

snippet     simultaneousLinearEquations
abbr        連立一次方程式
 const double EPS = 1e-8;
 typedef vector<double> vd;
 typedef vector<vd> mat;
 
 vd simultaneousLinearEquations(const mat &A, const vd &b){
     int n = A.size();
     mat B(n, vd(n + 1));
     rep(i,n) rep(j,n) B[i][j] = A[i][j];
     rep(i,n) B[i][n] = b[i];
 
     rep(i,n){
         int pivot = i;
         for(int j = i; j < n; j++){
             if(abs(B[i][j]) > abs(B[pivot][i])) pivot = j;
         }
         swap(B[i], B[pivot]);
 
         if(abs(B[i][i]) < EPS) return vd(); //解なし or 一意ではない
 
         for(int j = i + 1; j <= n; j++) B[i][j] /= B[i][i];
         rep(j,n){
             if(i != j){
                 for(int k = i + 1; k <= n; k++) B[j][k] -= B[j][i] * B[i][k];
             }
         }
     }
     vd x(n);
     rep(i,n) x[i] = B[i][n];
     return x;
 }


snippet     eulerPhi
abbr        オイラー関数
 const int MAX_N = 100;
 
 //オイラー関数の値を求める
 int eulerPhi(int n){
     int res = n;
     for(int i = 2; i * i <= n; i++){
         if(n % i == 0){
             res = res / i * (i - 1);
             for(; n % i == 0; n /= i);
         }
     }
     if(n != 1) res = res / n * (n - 1);
     return res;
 }
 
 int euler[MAX_N];
 
 //オイラー関数のテーブルを作る
 void eulerPhi2(){
     rep(i,MAX_N) euler[i] = i;
     for(int i = 2; i < MAX_N; i++){
         if(euler[i] == i){
             for(int j = i; j < MAX_N; j += i) euler[j] = euler[j] / i * (i - 1);
         }
     }
 }

snippet     repalceAll
abbr        文字列の置き換え
 string replaceAll(string s, string from, string to){
     vector<int> all;
     string tmp = s, tmp_space = s;
 
     string::size_type pos = tmp.find(from);
     while(pos != string::npos){
         all.emplace_back(pos);
         pos = tmp.find(from, pos + from.size());
     }
 
     //string space(from.size(), ' ');
     rep(j,all.size()){
         tmp.replace(all[j] + (to.size() - from.size()) * j, from.size(), to);
         //tmp_space.replace(all[j] + (from.size() - to.size()) * j, from.size(), space);
     }
     //if(tmp_space.find(to) != string::npos) return "0";
     if(s == tmp || all.empty()) "0";
 
     return tmp;
 }

snippet     JoinInterval
abbr        区間の結合
 //区間A[a,b]と区間B[c,d]の関係
 int intervalState(int a, int b, int c, int d){
     if(a < c && b < c) return 0;            //A < B
     else if(a > d && b > d) return 1;       //A > B
     else if(a <= c && d <= b) return 2;     //A -> B
     else if(c < a && b < d) return 3;       //B -> A
     else if(a <= c && b < d) return 4;      //A <= B
     else if(c < a && d <= b) return 5;      //A >= B
     return -1;
 }
 
 //Give input directly to vector<pair<int, int>> in
 vector<pair<int, int>> JoinInterval(vector<pair<int,int>> in){
     vector<pair<int, int>> v;
     rep(i,in.size()) in[i].second *= -1;
     sort(all(in));
     rep(i,in.size()) in[i].second *= -1;
 
     rep(i,in.size()){
         if(v.empty()) v.emplace_back(in[i]);
         else{
             pair<int, int> &u = v.back();
             int tmp = intervalState(in[i].first,in[i].second,u.first,u.second);
             switch (tmp){
                 case 0:
                 case 1:
                     v.emplace_back(in[i]);
                     break;
                 case 2:
                     u.first = in[i].first;
                     u.second = in[i].second;
                     break;
                 case 3:
                     break;
                 case 4:
                 case 5:
                     u.first = min(u.first, in[i].first);
                     u.second = max(u.second, in[i].second);
                     break;
                 case -1:
                     assert(0);
             }
         }
     }
     sort(all(v));
     return v;
 }

snippet     closedLoop
abbr        閉路の検出
 const int MAX_V = 505;
 
 vector<int> g[MAX_V];
 vector<int> tp;
 
 bool visit(int v, vector<int> &color){
     color[v] = 1;
     rep(i,g[v].size()){
         int d = g[v][i];
         if(color[d] == 2) continue;
         if(color[d] == 1) return false;
         if(not visit(d, color)) return false;
     }
     tp.emplace_back(v);
     color[v] = 2;
     return true;
 }
 
 bool topologicalSort(int v){
     vector<int> color(v);
     rep(i,v){
         if(not color[i] && not visit(i, color)) return false;
     }
     reverse(all(tp));
     return true;
 }

snippet     treeDP
abbr        全方位木DPによる木の直径の演算
 struct edge {
     int to, cost;
 };
 
 vector< edge > g[100000];
 long long dist[100000];
 
 
 void dfs1(int idx, int par) {
     for(edge &e : g[idx]) {
         if(e.to == par) continue;
         dfs1(e.to, idx);
         dist[idx] = max(dist[idx], dist[e.to] + e.cost);
     }
 }
 
 int dfs2(int idx, int d_par, int par) {
     vector< pair< int, int > > d_child;
     d_child.emplace_back(0, -1); // 番兵みたいなアレ
     for(edge &e : g[idx]) {
         if(e.to == par) d_child.emplace_back(d_par + e.cost, e.to);
         else d_child.emplace_back(e.cost + dist[e.to], e.to);
     }
     sort(d_child.rbegin(), d_child.rend());
     int ret = d_child[0].first + d_child[1].first; // 最大から 2 つ
     for(edge &e : g[idx]) {
         if(e.to == par) continue;
         // 基本は d_child() の最大が d_par になるが, e.to の部分木が最大値のときはそれを取り除く必要がある
         ret = max(ret, dfs2(e.to, d_child[d_child[0].second == e.to].first, idx));
     }
     return (ret);
 }
 
 int solve(int v/*頂点数*/){
     dfs1(v / 2, - 1);
     return dfs(v / 2, 0, - 1);
 }


snippet     maximumSubarray
abbr        連続した部分列の和の最大値を求める
 int maximumSubarray(vector<int> v){
     int sum = 0, ans = -INF;
     for(auto i : v){
         sum = max(i, sum + i);
         ans = max(ans, sum);
     }
     return ans;
 }

snippet     cumulativeSum2D
abbr        2次元累積和
 template <class T> class CumulativeSum2D {
     public:
         vector<vector<T>> s;
         int h, w;
         CumulativeSum2D(const vector<vector<T>> &a) {
             h = a.size();
             w = a[0].size();
             s = vector<vector<T>>(h + 1, vector<T>(w + 1, 0));
             rep(i, h) rep(j, w) s[i + 1][j + 1] = a[i][j];
             rep(i, h + 1) rep(j, w) s[i][j + 1] += s[i][j];
             rep(i, h) rep(j, w + 1) s[i + 1][j] += s[i][j];
         }
 
         int sum(int i, int j, int h, int w) {
             return s[i + h][j + w] - s[i][j + w] + s[i][j] - s[i + h][j];
         }
 
         int maximumSubMatrix(){
             int ret = -INF;
             rep(i,h + 1) rep(j,w + 1) range(k,1,h - i + 1) range(l,1,w - j + 1){
                 ret = max(ret, sum(i,j,k,l));
             }
             return ret;
         }
 };

snippet     maximumrectangule
abbr        最大長方形
 int maximumRectangule(int h, int w, char m[505][505]){
     int c[505][505] = {{0}};
     rep(i,w){
         int cnt = 0;
         rep(j,h){
             if(m[j][i] == '.'){
                 cnt++;
                 c[j][i] = cnt;
             }else{
                 cnt = 0;
             }
         }
     }
     rep(i,h) c[i][w] = 0;
 
     int ans = 0;
     rep(i,h){
         vector<pair<int, int>> v; //height, pos
         rep(j,w + 1){
             if(v.empty()) v.emplace_back(c[i][j],j);
             else{
                 if(v.back().first < c[i][j]) v.emplace_back(c[i][j],j);
                 else{
                     int target = j;
                     while(not v.empty() && v.back().first >= c[i][j]){
                         ans = max(ans, v.back().first * (j - v.back().second));
                         target = v.back().second;
                         v.pop_back();
                     }
                     v.emplace_back(c[i][j],target);
                 }
             }
         }
     }
     return ans;
 }

snippet     matrixChainMultiplication
abbr        連鎖行列積
 // O(n^3)
 // dp[i][j] := [i,j]を計算するための最小の掛け算の回数
 // i番目の行列の(行数、列数)が、(p[i - 1], p[i])
 int matrixChainMultiplication(int n, int p[105]){
     int dp[105][105] = {{0}};
     range(seg, 2, n + 1){ //計算する行列の個数
         range(i, 1, n - seg + 2){
             int j = i + seg - 1;
             dp[i][j] = INT_MAX;
             range(k,i,j){
                 int cost = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j];
                 dp[i][j] = min(dp[i][j], cost);
             }
         }
     }
     return dp[1][n];
 }

snippet     bitSubSet
abbr        部分集合のビット列を降順に列挙する
 int sub = s;
 do {
 
 	sub = (sub - 1) & s;
 }while(sub != s);

snippet     bitSizeSet
abbr        0 ~ n - 1 に含まれる、サイズkの部分集合を列挙する
 //0 ~ n - 1 に含まれる、サイズkの部分集合を列挙する
 int comb = (1 << k) - 1;
 while(comb < (1 << n)){
 	int x = comb & -comb, y = comb + x;
 	comb = ((comb & ~y) / x >> 1) | y;
 }

snippet     AhoCorasick
abbr        文字列を扱う木構造
 const int MAX = 26;
 const char OFFSET = 'a';
 
 struct Node{
 	int nxt[MAX+1];			// 次のalphabeteのノード番号
 	int exist;				// 子ども以下に存在する文字列の数の合計
 	vector<int> accept;		// その文字列id
 	Node() : exist(0){memset(nxt, -1, sizeof(nxt));}
 };
 
 class Trie{
 	private:
 		void update_direct(int node,int id){
 			nodes[node].accept.push_back(id);
 		}
 		void update_child(int node,int child,int id){
 			++nodes[node].exist;
 		}
 		void add(const string &str,int str_index,int node_index,int id){
 			if(str_index == str.size())
 				update_direct(node_index, id);
 			else{
 				const int c = str[str_index] - OFFSET;
 				if(nodes[node_index].nxt[c] == -1) {
 					nodes[node_index].nxt[c] = (int) nodes.size();
 					nodes.push_back(Node());
 				}
 				add(str, str_index + 1, nodes[node_index].nxt[c], id);
 				update_child(node_index, nodes[node_index].nxt[c], id);
 			}
 		}
 
 	public:
 		vector<Node>nodes;
 		int root;
 		Trie() : root(0){nodes.push_back(Node());}
 		void add(const string &str,int id){add(str, 0, 0, id);}
 		void add(const string &str){add(str, nodes[0].exist);}
 		int size(){return (nodes[0].exist);}
 		int nodesize(){return ((int) nodes.size());}
 };
 
 class AhoCorasick : public Trie{
 	public: 
 		static const int FAIL = MAX;
 		vector<int> correct;
 		AhoCorasick() : Trie() {}
 
 		void build(){
 			correct.resize(nodes.size());
 			rep(i,nodes.size())correct[i]=(int)nodes[i].accept.size();
 
 			queue<int> que;
 			rep(i,MAX+1){
 				if(~nodes[0].nxt[i]) {
 					nodes[nodes[0].nxt[i]].nxt[FAIL] = 0;
 					que.emplace(nodes[0].nxt[i]);
 				}else nodes[0].nxt[i] = 0;
 			}
 			while(!que.empty()) {
 				Node now = nodes[que.front()];
 				correct[que.front()] += correct[now.nxt[FAIL]];
 				que.pop();
 				rep(i,MAX){
 					if(now.nxt[i] == -1) continue;
 					int fail = now.nxt[FAIL];
 					while(nodes[fail].nxt[i] == -1) {
 						fail = nodes[fail].nxt[FAIL];
 					}
 					nodes[now.nxt[i]].nxt[FAIL] = nodes[fail].nxt[i];
 
 					auto &u = nodes[now.nxt[i]].accept;
 					auto &v = nodes[nodes[fail].nxt[i]].accept;
 					vector<int> accept;
 					set_union(all(u),all(v),back_inserter(accept));
 					u=accept;
 					que.emplace(now.nxt[i]);
 				}
 			}
 		}
 		int match(const string &str,vector<int> &result,int now=0){
 			result.assign(size(),0);
 			int count=0;
 			for(auto &c:str) {
 				while(nodes[now].nxt[c-OFFSET]==-1)now=nodes[now].nxt[FAIL];
 				now = nodes[now].nxt[c-OFFSET];
 				count += correct[now];
 				for(auto &v:nodes[now].accept)result[v]++;
 			}
 			return count;
 		}
 		int next(int now,char c){
 			while(nodes[now].nxt[c-OFFSET]==-1)now=nodes[now].nxt[FAIL];
 			return nodes[now].nxt[c-OFFSET];
 		}
 };


snippet     eulerGraph
abbr        オイラー路の構築
 class EulerGraph{
 	private:
 		bool directed;
 		bool isList;
 		int degreeCheck(int a){
 			if(a == 0) return EULER_GRAPH;
 			else if(a == 2) return SEMI_EULER_GRAPH;
 			else return 0;
 		}
 		int degreeCheck(vector<vector<int>> d){
 			int s = 0, t = 0, p = 0;
 			for(auto i : d){
 				if(i[0] == i[1] + 1) s++;
 				else if(i[0] + 1 == i[1]) t++;
 				else if(i[0] == i[1]) p++;
 				else return 0;
 			}
 			if(s == 0 && t == 0) return EULER_GRAPH;
 			else if(s == 1 && t == 1) return SEMI_EULER_GRAPH;
 			else return 0;
 		}
 		int checkEulerDirectedList(){
 			vector<vector<int>> degree(g.size(), vector<int>(2,0)); //0:out, 1:in
 			rep(i,g.size()){
 				degree[i][0] = g[i].size();
 				for(auto j : g[i]){
 					degree[j][1]++;
 				}
 			}
 			return degreeCheck(degree);
 		}
 		int checkEulerIndirectedList(){
 			int a = 0;
 			rep(i,g.size()){
 				if(g[i].size() % 2) a++;
 			}
 			return degreeCheck(a);
 		}
 		int checkEulerDirectedMat(){
 			vector<vector<int>> degree(g.size(), vector<int>(2,0)); //0:out, 1:in
 			rep(i,g.size()){
 				rep(j,g[i].size()){
 					if(g[i][j] > 0){
 						degree[i][0]++;
 						degree[j][1]++;
 					}
 				}
 			}
 			return degreeCheck(degree);
 		}
 		int checkEulerIndirectedMat(){
 			int a = 0;
 			rep(i,g.size()){
 				int degree = 0;
 				rep(j,g[i].size()){
 					if(g[i][j] > 0) degree++;
 				}
 				if(degree % 2) a++;
 			}
 			return degreeCheck(a);
 		}
 		//gが隣接リスト
 		//gを破壊する
 		//始点、隣接リスト、有向辺かどうか
 		vector<int> ListEulerianTrail(const int s, const bool directed) {
 			function<void (int, vector<int> &)> dfs = [&](int u, vector<int> &trail) {
 				while (not g[u].empty()) {
 					int v = g[u].back();
 					g[u].pop_back();
 					if (not directed) {
 						for (int i = 0; i < g[v].size(); i ++) {
 							if (g[v][i] == u) {
 								g[v].erase(g[v].begin() + i);
 								break;
 							}
 						}
 					}
 					dfs(v, trail);
 				}
 				trail.emplace_back(u);
 			};
 			vector<int> trail;
 			dfs(s, trail);
 			reverse(trail.begin(), trail.end());
 			return trail;
 		}
 		//gが隣接行列
 		//gを破壊する
 		vector<int> MatEulerianTrail(const int s, const bool directed) {
 			function<void (int, vector<int> &)> dfs = [&](int u, vector<int> &trail) {
 				for (int v = 0; v < g.size(); v ++) if (g[u][v] > 0) {
 					g[u][v]--;
 					if (not directed) g[v][u]--;
 					dfs(v, trail);
 				}
 				trail.push_back(u);
 			};
 			vector<int> trail;
 			dfs(s, trail);
 			reverse(trail.begin(), trail.end());
 			return trail;
 		}
 	public:
 		vector<vector<int>> g;
 		EulerGraph(vector<vector<int>> &g, bool directed, bool isList){
 			this->g = g;
 			this->directed = directed;
 			this->isList = isList;
 		}
 
 		static const int EULER_GRAPH = 1;
 		static const int SEMI_EULER_GRAPH = 2;
 		int isEulerGraph(){
 			if(isList){
 				if(directed) return checkEulerDirectedList();
 				else return checkEulerIndirectedList();
 			}else{
 				if(directed) return checkEulerDirectedMat();
 				else return checkEulerIndirectedMat();
 			}
 			return -1;
 		}
 		vector<int> getEulerianTrail(const int s){
 			if(isList) return ListEulerianTrail(s,directed);
 			else return MatEulerianTrail(s, directed);
 		}
 };

snippet     grundy
abbr        グランディー数を求める再帰関数
 int memo[101];
 int grundy(int n){
 	auto& p = memo[n];
 	if(p != -1) return p;
 
 	set<int> s;
 
 
 	int res = 0;
 	while(s.count(res)) res++;
 	return p = res;
 }

snippet     dijkstra-extend
abbr        拡張dijkstra
 //https://beta.atcoder.jp/contests/code-thanks-festival-2015-open/tasks/code_thanks_festival_2015_g
 
 class Edge{ //頂点、コスト、辺に与えられた識別番
 	public:
 		int to, cost, color;
 		Edge(int to, int cost, int color) : to(to) ,cost(cost) ,color(color) {}
 };
 
 class Node{
 	public:
 		int dis;
 		bool isUsed;
 		Node(){
 			this->dis = INF;
 			this->isUsed = 0;
 		}
 };
 
 class Dijkstra{
 	private:
 		vector<vector<Edge>> g;
 		vector<map<int, Node>> node; // node[i][j] := 頂点iに色jで来たとき
 		int n;
 	public:
 		Dijkstra(int n){
 			this->n = n;
 			g = vector<vector<Edge>>(n);
 			node = vector<map<int, Node>>(n);
 		}
 		int solve(int start, int goal, int spre){	//spre スタート時の状態。何もなければ-1
 			typedef pair<int,pair<int, int>> ppi;
 			priority_queue<ppi, vector<ppi>, greater<ppi>> q;
 
 			q.push(make_pair(0, make_pair(start, spre)));
 			node[start][spre].dis = 0;
 
 			ppi u;
 			while(not q.empty()){
 				u = q.top(); q.pop();
 				int cur = u.second.first;
 				int pre = u.second.second;
 				node[cur][pre].isUsed = 1;
 				if(cur == goal) break;//これがないとTLE
 				//breakしないとgoalに到着したコストを各色について求めてしまう
 				//最後の色の指定がないので、どんな色でもいいから最短経路が決まったらbreakする。
 
 				for(const auto& e : g[cur]){
 					int to = e.to;
 					int color = e.color;
 					int cost = e.cost + abs(pre - color) + u.first;
 
 					auto& next = node[to][color];
 					if(next.isUsed == 0){
 						if(not node[to].count(color) || next.dis > cost){
 							next.dis = cost;
 							q.push(make_pair(next.dis, make_pair(to, color)));
 						}
 					}
 				}
 			}
 
 			int mini = INF;
 			for(auto it : node[goal]){
 				mini = min(mini, it.second.dis);
 			}
 			return mini;
 		}
 		void addEdge(int from, int to, int cost, int color){
 			g[from].emplace_back(Edge{to, cost, color});
 			g[to].emplace_back(Edge{from, cost, color});
 		}
 };

snippet     rightHand
abbr        右手法
 // 右手法
 // right[d] := d方向（東から時計回り）を向いているときの優先順序
 const int rightHand[4][4] = {
 	{1,0,3,2},
 	{2,1,0,3},
 	{3,2,1,0},
 	{0,3,2,1}
 };
#bigInteger
# 配列を用いた整数の表現
#combination
# 組み合わせ、べき乗、階乗

snippet     combination
abbr        組み合わせ、べき乗、階乗
 const long long M = 1000000007;
 
 class Combination{
 	public:
 		vector<long long> fact, rev;
 		Combination(int n = 200005) : fact(n), rev(n) { // n = h + w
 			fact[0] = fact[1] = 1;
 			range(i,2,n){
 				(fact[i] = fact[i - 1] * i) %= M;
 			}
 			rev[n - 1] = power(fact[n - 1], M - 2) % M;
 			for (int i = n - 1; i > 0; i--) {
 				rev[i - 1] = rev[i] * i % M;
 			}
 		}
 		long long power(long long x, long long n){ //べき乗 x^n mod M
 			long long res = 1;
 			if(n > 0){
 				res = power(x, n / 2);
 				if(n % 2 == 0) res = (res * res) % M;
 				else res = (((res * res) % M) * x ) % M;
 			}
 			return res;
 		}
 		long long factorial(long long l, long long r){ return fact[r] * rev[l - 1] % M; }
 		long long factorial(long long n){ return fact[n]; }
 		long long combination(long long n, long long r){ //nCr (1,1)から(w,h)だと、引数は(w - 1, h - 1, M)
 			return factorial(n - r + 1, n) * rev[r] % M;
 		}
 };

snippet     cumulativeSum
abbr        累積和
    template<typename T>
    class CumulativeSum {
            vector<T> a;
        public:
            CumulativeSum(vector<T>& x) : a(x.size() + 1,0) {
                rep(i,x.size()){
                    a[i + 1] = a[i] + x[i];
                }
            }
            T get(int l, int r){ // [l, r]の区間の合計を求める
                return a[r + 1] - a[l];
            }
            T operator [] (const int i) const {
                return a[i];
            }
    };

snippet     bigInteger
abbr        配列を用いた整数の表現
 class Number{
 	int n;
 	public:
 		vector<int> a;
 		Number(int n) : n(n), a(n, 0) { }
 		void add(vector<int>& b){
 			reverse(all(b));
 			rep(i,b.size()){
 				a[i + 1] += (a[i] + b[i]) / 10;
 				a[i] = (a[i] + b[i]) % 10;
 			}
 			rep(i,n - 1){
 				a[i + 1] += a[i] / 10;
 				a[i] = a[i] % 10;
 			}
 		}
 		void add(string s){
 			vector<int> b(s.size());
 			rep(i,s.size()) b[i] = s[i] - '0';
 			add(b);
 		}
 		vector<int> get(){
 			vector<int> res = a;
 			while(res.back() == 0 and not res.empty()) res.pop_back();
 			reverse(all(res));
 			return res;
 		}
        void out(){
            vector<int> res = get();
            for(auto i : res) cout << i;
            cout << endl;
        }
 };
#slideMinimum
# スライド最小値
#patienceSort
# 数列を単調増加（減少）となるような部分列に分割する。
#ulamSpiral
# ウラムの螺旋
#compressCoordinate
# 座標圧縮

snippet     compressCoordinate
abbr        座標圧縮
 template <typename T>
 class CompressCoordinate{
 	public:
 		vector<T> a, c; // 圧縮した数列、ソートした数列
 		CompressCoordinate(const vector<T>& arg){
 			a = vector<T>(arg.size());
 			c = arg;
 			compress(c);
 			rep(i,arg.size()){
 				a[i] = lb(c, arg[i]);
 			}
 		}
 		void compress(vector<T> &v) {
 			sort(v.begin(), v.end());
 			v.erase(unique(v.begin(),v.end()),v.end());
 		}
 		int lb(const vector<T>& v, T num){
 			return lower_bound(all(v), num) - v.begin();
 		}
 		int map(T a){
 			return lb(c, a);
 		}
 };

snippet     ulamSpiral
abbr        ウラムの螺旋
 class UlamSpiral{
 	private:
 		const static int MAX_N = 2005;
 		const static int kN = 1000005;
 		void primeNumber(bool prime[kN]){
 			rep(i,kN) prime[i] = 1;
 			prime[0] = prime[1] = 0;
 			rep(i,kN){
 				if(prime[i]){
 					for(int j = i + i; j < kN; j+=i){
 						prime[j] = 0;
 					}
 				}
 			}
 		}
 
 	public:
 		int spiral[MAX_N][MAX_N] = {{0}};
 		int CENTER;
 		bool p[kN] = {0};
 		const int dy[4] = { 0,-1, 0, 1}; //反時計回り
 		const int dx[4] = { 1, 0,-1, 0};
 		UlamSpiral(int n){ //自然数の数
 			CENTER = MAX_N / 2;;
 
 			primeNumber(p);
 
 			int x = CENTER, y = CENTER;
 			int dir = 0; //右向き
 			rep(i,n){
 				spiral[y][x] = i + 1;
 				x += dx[dir % 4];
 				y += dy[dir % 4];
 
 				int nx = x + dx[(dir + 1) % 4];
 				int ny = y + dy[(dir + 1) % 4];
 				if(spiral[ny][nx] == 0){
 					dir++;
 				}
 			}
 		}
 
 		pair<int, int> getPoint(int n){ //数値nがある座標
 			rep(i,MAX_N){
 				rep(j,MAX_N){
 					if(spiral[i][j] == n){
 						return make_pair(i,j);
 					}
 				}
 			}
 			assert(false);
 			return make_pair(-1,-1);
 		}
 
 		bool isPrime(int n){
 			return p[n];
 		}
 		bool isPrime(int y, int x){
 			if(spiral[y][x] == -1) return 0;
 			return p[spiral[y][x]];
 		}
 
 		void out(){
 			show(CENTER)
 			for(int i = CENTER - 11; i < CENTER + 11; i++){
 			for(int j = CENTER - 11; j < CENTER + 11; j++){
 				//printf("%02d ", spiral[i][j]);
 				if(p[spiral[i][j]]){
 				printf("%02d ", spiral[i][j]);
 				}else{
 					cout << "__ ";
 				}
 			}
 			cout << endl;
 
 			}
 		}
 };

snippet     patienceSort
abbr        数列を単調増加（減少）となるような部分列に分割する。
 template < typename pile_type >
 struct pile_less : std::binary_function<pile_type, pile_type, bool> {
 	bool operator()(const pile_type& x, const pile_type& y) const {
 		return x.back() < y.back();
 	}
 };
 
 template < typename pile_type >
 struct pile_more : std::binary_function<pile_type, pile_type, bool> {
 	bool operator()(const pile_type& x, const pile_type& y) const {
 		return x.back() > y.back();
 	}
 };
 
 template < typename PileType, typename TableType, typename InputIterator >
 void patience_sort(TableType& table, InputIterator first, InputIterator last) {
 	typedef PileType pile_type;
 	//typedef typename PileType::value_type card_type;
 	typedef TableType table_type;
 	typedef typename table_type::iterator iterator;
 
 	while (first != last) {
 		pile_type new_pile{*first};
 
 		// upper_bound	:	i < j -> a_i < a_j
 		// lower_bound	:	i < j -> a_i <= a_j
 		// pile_more	:	単調増加
 		// pile_less	:	単調減少
 		iterator pile_p = std::upper_bound(
 				table.begin(), table.end(), new_pile,
 				pile_more<pile_type>() );
 		if (pile_p != table.end()) {
 			pile_p->push_back(new_pile.back());
 		} else {
 			table.push_back(new_pile);
 		}
 		first++;
 	}
 }
 
 vector<vector<int>> psort(vector<int>& a){
 	vector<vector<int>> table;
 	patience_sort<vector<int>, vector<vector<int>>, vector<int>::const_iterator>(table, a.begin(), a.end());
 	return table;
 }

snippet     slideMinimum
abbr        スライド最小値
 // 幅がkである区間から最小値を求める
 template <class T>
 std::vector<int> slideMinimum(const std::vector<T> &a, int k) {
 	int n = a.size();
 	std::deque<int> deq;
 	std::vector<int> res;
 	for (int i = 0; i < n; i++) {
 		while (deq.size() && a[deq.back()] >= a[i]) deq.pop_back();
 		deq.push_back(i);
 		if (i - k + 1 >= 0) {
 			res.push_back(deq.front());
 			if (deq.front() == i - k + 1) deq.pop_front();
 		}
 	}
 	return res;
 }
#suffixArray
# 接尾辞配列による文字列検索
#rollingHash
# 文字列の検索。区間[l,r)のハッシュ値をO(1)で計算
#bakerBird
# O(総文字数 + 検索パターン文字数 * 10) 二次元パターン検索
#ahoCorasick
# O(s.size() + t.size()) 複数パターン検索
#knutMorrisPratt
# O(s.size() + t.size()) パターン検索

snippet     knutMorrisPratt
abbr        O(s.size() + t.size()) パターン検索
 class KnuthMorrisPratt{
 public:
 	vector<int> fail;
 	string t; // 探す文字列
 	KnuthMorrisPratt(string& t) {
 		this->t = t;
 		int m = t.size();
 		fail = vector<int>(m + 1);
 		int j = fail[0] = -1;
 		for (int i = 1; i <= m; ++i) {
 			while (j >= 0 && t[j] != t[i - 1]) j = fail[j];
 			fail[i] = ++j;
 		}
 	}
 	vector<int> match(string& s){ // s に含まれる連続する部分文字列 t のインデックスを返す
 		int n = s.size(), m = t.size();
 		vector<int> res;
 		for (int i = 0, k = 0; i < n; ++i) {
 			while (k >= 0 && t[k] != s[i]) k = fail[k];
 			if (++k >= m) {
 				res.emplace_back(i - m + 1); // match at s[i-m+1 .. i]
 				k = fail[k];
 			}
 		}
 		return res;
 	}
 };

snippet     ahoCorasick
abbr        O(s.size() + t.size()) 複数パターン検索
 const int MAX = 62;
 
 int encode(char c){
    if(isdigit(c)){
    	return  c - '0';
    }else if(islower(c)){
    	return 10 + c - 'a';
    }else if(isupper(c)){
    	return 10 + 26 + c - 'A';
    }
 
    assert(false && "invalid string");
 }
 
 struct Node{
 int nxt[MAX+1];			// 次のalphabeteのノード番号
 int exist;				// 子ども以下に存在する文字列の数の合計
 vector<int> accept;		// その文字列id
 Node() : exist(0){memset(nxt, -1, sizeof(nxt));}
 };
 
 class Trie{
    private:
    	void updateDirect(int node,int id){
    		ac.emplace_back(node);
    		nodes[node].accept.emplace_back(id);
    	}
    	void updateChild(int node,int child,int id){
    		++nodes[node].exist;
    	}
    	void add(const string &str,int str_index,int node_index,int id){
    		if(str_index == str.size())
    			updateDirect(node_index, id);
    		else{
    			const int c = encode(str[str_index]);
    			if(nodes[node_index].nxt[c] == -1) {
    				nodes[node_index].nxt[c] = (int) nodes.size();
    				nodes.emplace_back(Node());
    			}
    			add(str, str_index + 1, nodes[node_index].nxt[c], id);
    			updateChild(node_index, nodes[node_index].nxt[c], id);
    		}
    	}
    	void add(const string &str,int id){add(str, 0, 0, id);}
    public:
    	vector<Node>nodes;
    	vector<int> ac; // ac[i] := i番目のパターンを受理する状態番号
    	int root;
    	Trie() : root(0){nodes.emplace_back(Node());}
    	void add(const string &str){add(str, nodes[0].exist);}
    	int size(){return (nodes[0].exist);}
    	int nodesize(){return ((int) nodes.size());}
 };
 
 class AhoCorasick : public Trie{
 public: 
 	static const int FAIL = MAX;
 	vector<int> correct;
 	AhoCorasick() : Trie() {}
 
 	void build(){
 		correct.resize(nodes.size());
 		rep(i,nodes.size())correct[i]=(int)nodes[i].accept.size();
 
 		queue<int> que;
 		rep(i,MAX+1){
 			if(~nodes[0].nxt[i]) {
 				nodes[nodes[0].nxt[i]].nxt[FAIL] = 0;
 				que.emplace(nodes[0].nxt[i]);
 			}else nodes[0].nxt[i] = 0;
 		}
 		while(!que.empty()) {
 			Node now = nodes[que.front()];
 			correct[que.front()] += correct[now.nxt[FAIL]];
 			que.pop();
 			rep(i,MAX){
 				if(now.nxt[i] == -1) continue;
 				int fail = now.nxt[FAIL];
 				while(nodes[fail].nxt[i] == -1) {
 					fail = nodes[fail].nxt[FAIL];
 				}
 				nodes[now.nxt[i]].nxt[FAIL] = nodes[fail].nxt[i];
 
 				auto &u = nodes[now.nxt[i]].accept;
 				auto &v = nodes[nodes[fail].nxt[i]].accept;
 				vector<int> accept;
 				set_union(all(u),all(v),back_inserter(accept));
 				u=accept;
 				que.emplace(now.nxt[i]);
 			}
 		}
 	}
 	// result := 各パターンがそれぞれ何度マッチしたか
 	int match(const string &str,vector<int> &result,int now=0){
 		result.assign(size(),0);
 		int count=0;
 		for(auto &c:str) {
 			while(nodes[now].nxt[encode(c)]==-1)now=nodes[now].nxt[FAIL];
 			now = nodes[now].nxt[encode(c)];
 			count += correct[now];
 			for(auto &v:nodes[now].accept)result[v]++;
 		}
 		return count;
 	}
 	int next(int now,char c){
 		while(nodes[now].nxt[encode(c)]==-1)now=nodes[now].nxt[FAIL];
 		return nodes[now].nxt[encode(c)];
 	}
 };

snippet     bakerBird
abbr        O(総文字数 + 検索パターン文字数 * 10) 二次元パターン検索
 const int MAX = 62;
 
 int encode(char c){
 if(isdigit(c)){
 	return  c - '0';
 }else if(islower(c)){
 	return 10 + c - 'a';
 }else if(isupper(c)){
 	return 10 + 26 + c - 'A';
 }
 
 assert(false && "invalid string");
 }
 
 struct Node{
 int nxt[MAX+1];			// 次のalphabeteのノード番号
 int exist;				// 子ども以下に存在する文字列の数の合計
 vector<int> accept;		// その文字列id
 Node() : exist(0){memset(nxt, -1, sizeof(nxt));}
 };
 
 class Trie{
 private:
 	void updateDirect(int node,int id){
 		ac.emplace_back(node);
 		nodes[node].accept.emplace_back(id);
 	}
 	void updateChild(int node,int child,int id){
 		++nodes[node].exist;
 	}
 	void add(const string &str,int str_index,int node_index,int id){
 		if(str_index == str.size())
 			updateDirect(node_index, id);
 		else{
 			const int c = encode(str[str_index]);
 			if(nodes[node_index].nxt[c] == -1) {
 				nodes[node_index].nxt[c] = (int) nodes.size();
 				nodes.emplace_back(Node());
 			}
 			add(str, str_index + 1, nodes[node_index].nxt[c], id);
 			updateChild(node_index, nodes[node_index].nxt[c], id);
 		}
 	}
 	void add(const string &str,int id){add(str, 0, 0, id);}
 public:
 	vector<Node>nodes;
 	vector<int> ac; // ac[i] := i番目のパターンを受理する状態番号
 	int root;
 	Trie() : root(0){nodes.emplace_back(Node());}
 	void add(const string &str){add(str, nodes[0].exist);}
 	int size(){return (nodes[0].exist);}
 	int nodesize(){return ((int) nodes.size());}
 };
 
 class AhoCorasick : public Trie{
 public: 
 	static const int FAIL = MAX;
 	vector<int> correct;
 	AhoCorasick() : Trie() {}
 
 	void build(){
 		correct.resize(nodes.size());
 		rep(i,nodes.size())correct[i]=(int)nodes[i].accept.size();
 
 		queue<int> que;
 		rep(i,MAX+1){
 			if(~nodes[0].nxt[i]) {
 				nodes[nodes[0].nxt[i]].nxt[FAIL] = 0;
 				que.emplace(nodes[0].nxt[i]);
 			}else nodes[0].nxt[i] = 0;
 		}
 		while(!que.empty()) {
 			Node now = nodes[que.front()];
 			correct[que.front()] += correct[now.nxt[FAIL]];
 			que.pop();
 			rep(i,MAX){
 				if(now.nxt[i] == -1) continue;
 				int fail = now.nxt[FAIL];
 				while(nodes[fail].nxt[i] == -1) {
 					fail = nodes[fail].nxt[FAIL];
 				}
 				nodes[now.nxt[i]].nxt[FAIL] = nodes[fail].nxt[i];
 
 				auto &u = nodes[now.nxt[i]].accept;
 				auto &v = nodes[nodes[fail].nxt[i]].accept;
 				vector<int> accept;
 				set_union(all(u),all(v),back_inserter(accept));
 				u=accept;
 				que.emplace(now.nxt[i]);
 			}
 		}
 	}
 	// result := 各パターンがそれぞれ何度マッチしたか
 	int match(const string &str,vector<int> &result,int now=0){
 		result.assign(size(),0);
 		int count=0;
 		for(auto &c:str) {
 			while(nodes[now].nxt[encode(c)]==-1)now=nodes[now].nxt[FAIL];
 			now = nodes[now].nxt[encode(c)];
 			count += correct[now];
 			for(auto &v:nodes[now].accept)result[v]++;
 		}
 		return count;
 	}
 	int next(int now,char c){
 		while(nodes[now].nxt[encode(c)]==-1)now=nodes[now].nxt[FAIL];
 		return nodes[now].nxt[encode(c)];
 	}
 };
 
 class KnuthMorrisPratt{
 public:
 	vector<int> fail;
 	string t; // 探す文字列
 	KnuthMorrisPratt(string& t) {
 		this->t = t;
 		int m = t.size();
 		fail = vector<int>(m + 1);
 		int j = fail[0] = -1;
 		for (int i = 1; i <= m; ++i) {
 			while (j >= 0 && t[j] != t[i - 1]) j = fail[j];
 			fail[i] = ++j;
 		}
 	}
 	vector<int> match(string& s){ // s に含まれる連続する部分文字列 t のインデックスを返す
 		int n = s.size(), m = t.size();
 		vector<int> res;
 		for (int i = 0, k = 0; i < n; ++i) {
 			while (k >= 0 && t[k] != s[i]) k = fail[k];
 			if (++k >= m) {
 				res.emplace_back(i - m + 1); // match at s[i-m+1 .. i]
 				k = fail[k];
 			}
 		}
 		return res;
 	}
 };
 
 // tに一致するsの部分集合の左上の座標を返す
 // O(N + |A|M) N:=総文字数 M:=パターンの文字数 |A|:=アルファベット+数字集合のサイズ
 vector<pair<int, int>> BakerBird(vector<string>& s, vector<string>& t){
 AhoCorasick aho;
 for(auto str : t) aho.add(str);
 aho.build();
 
 vector<vector<int>> status(s.size(), vector<int>(s[0].size()));
 rep(i,s.size()){
 	int now = 0;
 	rep(j,s[0].size()){
 		now = aho.next(now, s[i][j]);
 		status[i][j] = now;
 	}
 }
 
 string patern;
 rep(i,aho.ac.size()){
 	patern += aho.ac[i] + '0';
 }
 
 vector<pair<int, int>> res;
 KnuthMorrisPratt kmp(patern);
 range(i,t[0].size() - 1, s[0].size()){
 	string sstr; // statusを縦に見た文字列
 	rep(j,s.size()){
 		sstr += status[j][i] + '0';
 	}
 	for(auto y : kmp.match(sstr)){
 		res.emplace_back(y, i - t[0].size() + 1);
 	}
 }
 sort(all(res));
 return res;
 }

snippet     rollingHash
abbr        文字列の検索。区間[l,r)のハッシュ値をO(1)で計算
 class RollingHash {
 	public:
 		typedef pair<long long,long long> hash_type;
 		long long base1, base2;
 		long long mod1, mod2;
 		vector<long long> hash1, hash2;
 		vector<long long> pow1, pow2;
 
 		RollingHash(const string &s) : base1(1009), base2(1007), mod1(1000000007), mod2(1000000009) {
 			int n = s.size();
 
 			hash1.assign(n + 1,0);
 			hash2.assign(n + 1,0);
 			pow1.assign(n + 1,1);
 			pow2.assign(n + 1,1);
 
 			rep(i,n){
 				hash1[i + 1] = (hash1[i] + s[i]) * base1 % mod1;
 				hash2[i + 1] = (hash2[i] + s[i]) * base2 % mod2;
 				pow1[i + 1] = pow1[i] * base1 % mod1;
 				pow2[i + 1] = pow2[i] * base2 % mod2;
 			}
 		}
 
 		hash_type get(int l, int r) { // 区間[l,r)のハッシュ値を計算する
 			long long t1 = ((hash1[r] - hash1[l] * pow1[r - l]) % mod1 + mod1) % mod1;
 			long long t2 = ((hash2[r] - hash2[l] * pow2[r - l]) % mod2 + mod2) % mod2;
 			return make_pair(t1, t2);
 		}
 
 		hash_type concat(hash_type h1, hash_type h2, int h2_len) {
 			return make_pair((h1.first * pow1[h2_len] + h2.first) % mod1, (h1.second * pow2[h2_len] + h2.second) % mod2);
 		}
 };

snippet     suffixArray
abbr        接尾辞配列による文字列検索
 class SuffixArray{
 	private:
 		int n;
 		vector<int> rank;
 	public:
 		string s;
 		vector<int> idx;
 		SuffixArray(string& s) : n(s.size()), rank(s.size() + 1), s(s), idx(s.size() + 1) {
 			rep(i,n + 1){
 				idx[i] = i;
 				rank[i] = i < n ? s[i] : -1;
 			}
 
 			int k;
 			auto comp = [&](const int i, const int j) -> bool {
 				if(rank[i] != rank[j]) return rank[i] < rank[j];
 				else{
 					int ri = i + k <= n ? rank[i + k] : -1;
 					int rj = j + k <= n ? rank[j + k] : -1;
 					return ri < rj;
 				}
 			};
 
 			vector<int> tmp(n + 1);
 			for (k = 1; k <= n; k*=2) {
 				sort(all(idx), comp);
 
 				tmp[idx[0]] = 0;
 				range(i,1,n + 1){
 					tmp[idx[i]] = tmp[idx[i - 1]] + (comp(idx[i - 1], idx[i]) ? 1 : 0);
 				}
 				rank = tmp;
 			}
 		}
 		bool contain(string& t){
 			int right = n, left = 0;
 			while(right - left > 1){
 				int mid = (right + left) / 2;
 				if(s.substr(idx[mid], t.size()) < t) left = mid;
 				else right = mid;
 			}
 			return s.substr(idx[right], t.size()) == t;
 		}
 };
#epsCondition
# EPS付きの大小比較、一致判定
snippet     mod
abbr        10^9 + 7
options     head
    const int M = 1000000007;

snippet     inf
abbr        const int INF = 1e18;
options     head
    const int INF = 1e18;

snippet     cinup
abbr        cin高速化
options     head
    cin.tie(0);
    ios::sync_with_stdio(false);

snippet     upper_bound
abbr        二分探索
    upper_bound(all(${1}), ${2});${0}

snippet     lower_bound
abbr        二分探索
    lower_bound(all(${1}), ${2});${0}

snippet     dydx
abbr        座標移動の配列
options     head
    const int dy[16] = { 0,-1, 0, 1, 1,-1, 1,-1, 0,-2, 0, 2};
    const int dx[16] = { 1, 0,-1, 0, 1, 1,-1,-1, 2, 0,-2, 0};

snippet     memset
    memset(${1}, ${2}, sizeof($1));${0}

snippet     rephw
    rep(i,h){
        rep(j,w){
            ${0}
        }
    }

snippet     gridRange
abbr        gridの範囲外かの判定
options     head
    if(ny < 0 || ny >= h || nx < 0 || nx >= w) continue;

snippet     defineOperator
abbr        演算子の定義
options     head
  bool operator ${1} ( const ${2} &right ) const {
     ${0}
  }

snippet     definell
abbr        define int long long
options     head
    #define int long long

snippet     yesno
abbr        yesかnoの出力
options     head
    cout << (f ? "yes" : "no") << endl;

snippet     YesNo
abbr        YesかNoの出力
options     head
    cout << (f ? "Yes" : "No") << endl;

snippet     YESNO
abbr        YESかNOの出力
options     head
    cout << (f ? "YES" : "NO") << endl;

snippet     adjacent_find
abbr        隣接する要素で条件を満たす最初の要素を検索する
options     head
    auto f = [](${1} a, $1 b){${2}};
    adjacent_find(all(${3}), f);${0}

snippet     all_of
abbr        範囲の全ての要素が条件を満たすかを判定する
options     head
    
    auto f = [](${1} a, $1 b){${2}};
    all_of(all(${3}), f);${0}

snippet     any_of
abbr        範囲のいずれかの要素が条件を満たすかを判定する
options     head
    
    auto f = [](${1} a, $1 b){${2}};
    any_of(all(${3}), f);${0}

snippet     lamda
    auto f = [](${1} a, $1 b){${2}};

snippet     vectors
abbr        多次元vector
options     head

    template <typename X, typename T>
    auto vectors(X x, T a) {
    	return vector<T>(x, a);
    }
    
    template <typename X, typename Y, typename Z, typename... Zs>
    auto vectors(X x, Y y, Z z, Zs... zs) {
    	auto cont = vectors(y, z, zs...);
    	return vector<decltype(cont)>(x, cont);
    }

snippet     vets
abbr        二次元vector
    vector<vector<${1}>> ${2}(${3}, vector<$1>(${4}));${0}

snippet     epsCondition
abbr        EPS付きの大小比較、一致判定
 // r の(誤差付きの)符号に従って, -1, 0, 1 を返す.
 int sgn(const double& r){ return (r > EPS) - (r < -EPS); }
 // a, b の(誤差付きの)大小比較の結果に従って, -1, 0, 1 を返す.
 int sgn(const double& a, const double &b){ return sgn(a - b); }
 
 // a > 0 は sgn(a) > 0
 // a < b は sgn(a, b) < 0
 // a >= b は sgn(a, b) >= 0

snippet     vout
abbr        vectorを空白区切で出力
 rep(i,v.size()){
     if(i) cout << ' ';
     cout << v[i];
 }
 cout << endl;

snippet     prime
abbr        10e9前後の素数
    999999937
    1000000021
    1000010029
    1000100077

snippet     compress-bit-string
abbr        ビット列から，各ビットがいくつ連続しているかの数列を返す

	// 0100001011110 -> 1 1 4 1 1 4 1
	vector<int> compressBitString(const string& s){
		vector<int> p;
		int t = s.front() - '0';
		int cnt = 0;
		rep(i,s.size()){
			if(s[i] == t + '0'){
				cnt++;
			}else{
				p.emplace_back(cnt);
				cnt = 1;
				t ^= 1;
			}
		}
		if(cnt != 0){
			p.emplace_back(cnt);
		}
		return p;
	}
snippet     get-digit-sum
abbr        桁数の和を求める関数
    int getDigitSum(int a){
        int res = 0;
        while(a != 0){
            res += a % 10;
            a /= 10;
        }
        return res;
    }
