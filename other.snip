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
