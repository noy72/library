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
