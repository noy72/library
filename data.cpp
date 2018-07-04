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
