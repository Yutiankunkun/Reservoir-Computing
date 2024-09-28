#!/usr/bin/env python
# -*- coding utf-8 -*-

import numpy as np
import networkx as nx

# 恒等写像関数
def identity(x):
    return x

# 入力層→リザバー
class Input:
    # 入力結合重み行列Winの初期化
    def __init__(self, N_u, N_x, input_scale, seed, inflag):
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        if inflag == 0:
          #初期値の生成
          self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))
        else:
          #初期値を外部から読み込み(ランダムに生成しない)
          self.Win = np.loadtxt('ReCo-134-0000_DNS_W_IN-Matrix.txt')


    # 入力結合重み行列Winによる重みづけ
    def __call__(self, u):
        return np.dot(self.Win, u)

# リザバー間結合
class Reservoir:
    # リカレント結合重み行列Wの初期化
    def __init__(self, N_x, density, rho, activation_func, leaking_rate,
                 seed, wflag):
        '''
        param N_x: リザバーのノード数
        param density: ネットワークの結合密度
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: ノードの活性化関数
        param leaking_rate: leaky integratorモデルのリーク率
        param seed: 乱数の種
        param wflag: 初期値は生成or外部
        '''
        self.seed = seed

        if wflag == 0:
          #Wの初期化
          self.W = self.make_connection(N_x, density, rho)
        else:
          #Wの初期化（外部から読み込み）
          self.W = np.loadtxt('ReCo-134-0000_DNS_A_RE-Matrix.dat')

        self.x = np.zeros(N_x)  # リザバー状態ベクトルの初期化
        self.activation_func = activation_func
        self.alpha = leaking_rate

    # リカレント結合重み行列の生成
    def make_connection(self, N_x, density, rho):
        # Erdos-Renyiランダムグラフ
        m = int(N_x*(N_x-1)*density/2)  # 総結合数
        G = nx.gnm_random_graph(N_x, m, self.seed)

        # 行列への変換(結合構造のみ）
        connection = nx.to_numpy_array(G)
        W = np.asarray(connection)

        # 非ゼロ要素を一様分布に従う乱数として生成
        rec_scale = 1.0
        np.random.seed(seed=self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # スペクトル半径の計算
        eigv_list = np.linalg.eigh(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        # 指定のスペクトル半径rhoに合わせてスケーリング
        W *= rho / sp_radius

        return W

    # リザバー状態ベクトルの更新
    def __call__(self, x_in):
        '''
        param x_in: 更新前の状態ベクトル
        return: 更新後の状態ベクトル
        '''
        #self.x = self.x.reshape(-1, 1)
        self.x = (1.0 - self.alpha) * self.x \
                 + self.alpha * self.activation_func(np.dot(self.W, self.x) \
                 + x_in)
        return self.x

    # リザバー状態ベクトルの初期化
    def reset_reservoir_state(self):
        self.x *= 0.0

# リザバー層→出力層
class Output:
    # 出力結合重み行列の初期化
    def __init__(self, N_x, N_y, seed, outflag, lastflag):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param seed: 乱数の種
        param outflag: 初期値は生成or外部
        '''
        self.outflag = outflag
        self.lastflag = lastflag

        if outflag ==0:
          # Woutを正規分布に従う乱数, 生成
          np.random.seed(seed=seed)
          self.Wout = np.random.normal(size=(N_y, N_x))
        else:
          #外部から学習したものを読み込み
          self.Wout = np.loadtxt('ReCo-134-0000_DNS_W_OUTPUT_and_C_OUTPUT.dat')
          self.Wout = np.asarray(self.Wout)
          self.Wout = np.transpose(self.Wout)
        

    # 出力結合重み行列による重みづけ
    def __call__(self, x):
        '''
        param x: N_x次元のベクトル
        return: N_y次元のベクトル
        '''
        return np.dot(self.Wout, x)

    # 学習済みの出力結合重み行列を設定
    def setweight(self, Wout_opt):
        """
        param lastflag: 学習後のもの使うか、学習済使うか
        """

        self.Wout = Wout_opt
        # if lastflag == 0:
        #   #学習した結果のWout         
        #   self.Wout = Wout_opt
        # else: 
        #   #外部で学習したモノをテストで利用する。
        #   self.Wout = np.loadtxt('ReCo-134-0000_DNS_W_OUTPUT_and_C_OUTPUT.dat')
        #   self.Wout = np.transpose(self.Wout)

# 出力層 → リザバー層 フィードバック
class Feedback:
    # フィードバック結合重み行列の初期化
    def __init__(self, N_y, N_x, fb_scale, seed):
        '''
        param N_y: 出力次元
        param N_x: リザバーのノード数
        param fb_scale: フィードバックスケーリング
        param seed: 乱数の種
        '''
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        self.Wfb = np.random.uniform(-fb_scale, fb_scale, (N_x, N_y))

    # フィードバック結合重み行列による重みづけ
    def __call__(self, y):
        '''
        param y: N_y次元のベクトル
        return: N_x次元のベクトル
        '''
        return np.dot(self.Wfb, y)

# リッジ回帰（beta=0のときは線形回帰）
class Tikhonov:
    def __init__(self, N_x, N_y, beta):
        self.beta = beta
        self.X_XT = np.zeros((N_x, N_x))
        self.D_XT = np.zeros((N_y, N_x))
        self.N_x = N_x

    # 学習用の行列の更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, np.transpose(x))
        self.D_XT += np.dot(d, np.transpose(x))

    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self):
        X_pseudo_inv = np.linalg.inv(self.X_XT + self.beta * np.identity(self.N_x))
        Wout_opt = np.dot(self.D_XT, X_pseudo_inv)
        return Wout_opt

# エコーステートネットワーク
class ESN:
    def __init__(self, N_u, N_y, N_x , density, input_scale,
                 rho, activation_func, fb_scale ,
                 fb_seed, noise_level , leaking_rate,
                 output_func, inv_output_func,
                 classification, average_window, seed  ,
                 inflag, wflag, outflag , lastflag ):
        self.seed = seed
        self.inflag = inflag
        self.wflag = wflag
        self.outflag = outflag
        self.lastflag = lastflag

        self.Input = Input(N_u, N_x, input_scale, seed, inflag)
        self.Reservoir = Reservoir(N_x, density, rho, activation_func, 
                                   leaking_rate, seed, wflag)
        self.Output = Output(N_x, N_y, seed, outflag, lastflag)

        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.y_prev = np.zeros(N_y)
        self.output_func = output_func
        self.inv_output_func = inv_output_func
        self.classification = classification

        # 出力層からのリザバーへのフィードバックの有無(Noneにしてるから無)
        if fb_scale is None:
            self.Feedback = None
        else:
            self.Feedback = Feedback(N_y, N_x, fb_scale, fb_seed)

        # リザバーの状態更新おけるノイズの有無
        if noise_level is None:
            self.noise = None
        else:
            np.random.seed(seed)
            self.noise = np.random.uniform(-noise_level, noise_level, 
                                           (self.N_x, 1))

        # 分類問題か否か
        if classification:
            if average_window is None:
                raise ValueError('Window for time average is not given!')
            else:
                self.window = np.zeros((average_window, N_x))

    # バッチ学習
    def train(self, U, D, optimizer, trans_len):
        '''
        U: 教師データの入力, データ長×N_u
        D: 教師データの出力, データ長×N_y
        optimizer: 学習器
        trans_len: 過渡期の長さ
        return: 学習中のモデル出力, データ長×N_y
        '''
        train_len = len(U)
        if trans_len is None:
            trans_len = 0
        Y = []
        LOSS = []

        # 時間発展
        for n in range(train_len):
            x_in = self.Input(U[n])
            
            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back
            # ノイズ
            if self.noise is not None:
                x_in += self.noise
            # リザバー状態ベクトル
            x = self.Reservoir(x_in)
            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = np.append(self.window, x.reshape(1, -1),
                                        axis=0)
                self.window = np.delete(self.window, 0, 0)
                x = np.average(self.window, axis=0)
            # 目標値
            d = D[n]
            d = self.inv_output_func(d)
            # 学習器
            
            if n > trans_len:  # 過渡期を過ぎたら
                optimizer(d, x)
            # 学習前のモデル出力
            y = self.Output(x)
            Y.append(self.output_func(y))
            self.y_prev = d
            
            #学習がどこまで行われているか
            #if n%100 == 0 and n <= trans_len:
            #  print('ステップ数',n,'No')
            #if n%100 == 0 and n > trans_len:
            #  print('ステップ数',n,)
            #  #'二乗和誤差', np.mean((d - y) ** 2)

        # 学習済みの出力結合重み行列を設定

        self.Output.setweight(optimizer.get_Wout_opt())
      
        # モデル出力（学習前）
        return np.array(Y)

    # バッチ学習後の予測（自律系のフリーラン）
    def run(self, U):
        test_len = len(U)
        Y_pred = []
        y = U[0]

        # 時間発展
        for n in range(test_len):
            x_in = self.Input(y)

            
            # フィードバック結合(今fb is Noneなので実行されない)
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back
            

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)

            # 学習後のモデル出力
            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            y = y_pred
            self.y_prev = y

        return np.array(Y_pred)