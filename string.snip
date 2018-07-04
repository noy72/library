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
