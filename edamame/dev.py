import colorsys
import datetime
import random
from functools import lru_cache
from math import cos, pi, sin

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from scipy.spatial import Delaunay
from shapely.geometry import Point


def make_grid(shape, ngrid=40, xrange=(139.0, 139.8), yrange=(35.1, 35.7)):
    # DataAnalysisフォルダー内での相対位置
    gdf3 = gpd.read_file(shape)

    # 陸地全域をlandとする。
    # 小領域を融合してくれる便利な関数dissolve()
    land = gdf3[['geometry']].dissolve()

    # land.plot()


    # 陸地内の格子点だけにする。
    x = np.linspace(*xrange, ngrid)
    y = np.linspace(*yrange, ngrid)
    gX, gY = np.meshgrid(x,y)
    gX = gX.reshape(-1)
    gY = gY.reshape(-1)

    # EPSGは座標系らしい。
    points = gpd.GeoSeries([Point(x,y) for x,y in zip(gX,gY)]).set_crs(epsg=6668)

    # 関数の構造上、1つの領域に複数の点が含まれているかどうかを判定できない。
    # 1つの領域に対して1つの点が含まれているかどうかは判定できる。時間がかかるが、最初に一回だけ行えば良い処理なので黙認する。
    inside = np.array([land.contains(p) for p in points]).reshape(-1)
    # print(inside, sum(inside))
    gX = gX[inside]
    gY = gY[inside]
    return gX, gY



def perimeters(ax, shapefile):
    """
    shapefileを白地図として作画し、Artist形式で返す。
    """
    gdf2 = gpd.read_file(shapefile)
    # 行政区域の輪郭を黒で描画
    elems = []
    for region in gdf2.iloc:
        e = ax.plot(*region.geometry.exterior.xy, "#888", linewidth=0.5)
        elems += e
    return elems


def station_list(year=2020, target="OX", PATH=""):
    fn = f"{PATH}TM{year}0000.txt"
    # df = pd.read_csv(fn, encoding="ShiftJIS")
    df = pd.read_csv(fn, encoding="CP932")
    return df[df[f'{target}_測定有無'].notna()]


def station_info(pref=14, year=2020, target="OX", PATH=""):
    df = station_list(year=year, target=target, PATH=PATH)
    df = df[df["都道府県コード"]==pref]
    df2 = df[["国環研局番", "８文字名", "標高(m)"]]
    # Warningが出るが、問題ないらしい。
    df2.loc[:, "latitude"] = df.loc[:, '緯度_度'] + df.loc[:, '緯度_分']/60 + df.loc[:, '緯度_秒']/3600
    df2.loc[:, "longitude"] = df.loc[:, '経度_度'] + df.loc[:, '経度_分']/60 + df.loc[:, '経度_秒']/3600
    df2 = df2.rename(columns={"８文字名":"name",  "標高(m)":"altitude"})
    return df2.set_index("国環研局番").T.to_dict("dict")


def colorify(v, min=0, max=240):
    if v > max:
        v = max
    if v < min:
        v = min
    r, g, b =colorsys.hsv_to_rgb(2/3 - (v-min)/(max-min)*(2/3), 1.0, 1.0)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'



def gridfunc(target, h, dfs, stations):
    X = []
    Y = []
    Z = []
    for loc, value in dfs.items():
        if target in value.columns:
            X.append(stations[loc]["longitude"])
            Y.append(stations[loc]["latitude"])
            Z.append(value.iloc[h][target])
    # print(Z)
    # 3次元の散布図を内挿する
    # interpolator = interp.CloughTocher2DInterpolator(np.array([X,Y]).T, Z)
    interpolator = interp.LinearNDInterpolator(np.array([X,Y]).T, Z)
    return interpolator, X, Y, np.array(Z)


def gridfunc2D(target, h, dfs, stations):
    X = []
    Y = []
    Zx = []
    Zy = []
    for loc, value in dfs.items():
        if target in value.columns:
            X.append(stations[loc]["longitude"])
            Y.append(stations[loc]["latitude"])
            ws = value.iloc[h]["WS"]
            wd = value.iloc[h]["WD"]
            if wd < 1:
                print("WARN", wd)
            if wd in (17, 0):
                wd = random.randint(1,16)
            # 風向は風が吹いてくる方角、北風は南向きに吹く
            # ベクトルは吹く先を表現する。
            wx = -ws * sin(wd*pi/8)
            wy = -ws * cos(wd*pi/8)
            Zx.append(wx)
            Zy.append(wy)
    # 3次元の散布図を内挿する
    # interpolator = interp.CloughTocher2DInterpolator(np.array([X,Y]).T, Z)
    ipx = interp.LinearNDInterpolator(np.array([X,Y]).T, Zx)
    ipy = interp.LinearNDInterpolator(np.array([X,Y]).T, Zy)
    return ipx, ipy, X, Y, np.array(Zx), np.array(Zy)


def draw(target, h, ax, gX, gY, dfs, stations, vcolorify, shape):

    if target == "WS":


        # 1. グリッド点の処理

        ipx, ipy, sX, sY, sZx, sZy = gridfunc2D(target, h, dfs, stations)
        # gX, gYは県内のグリッド点、ipx, ipyは内挿関数
        Zx = ipx(gX,gY)
        Zy = ipy(gX,gY)
        # 内挿できなかった格子点を除外する。
        X = gX[~np.isnan(Zy)]
        Y = gY[~np.isnan(Zy)]
        Zx = Zx[~np.isnan(Zy)]
        Zy = Zy[~np.isnan(Zy)]

        elems = []

        ws = (Zx**2+Zy**2)**0.5
        # print(ws)
        # print(ws[:5])
        colors = vcolorify(ws)
        # print(colors)

        # print(X[:1], Y[:1], Zx[:1], Zy[:1])
        # print(X,Y,Zx,Zy,colors)
        e = ax.quiver(X, Y, Zx, Zy, color=colors)


        # 2. 測定局の処理

        # print(np.isnan(sZx).sum())

        # Compute Delaunay
        points = np.array([sX,sY]).T
        # 欠測を除く。
        number = ~np.isnan(sZx)
        points = points[number]
        sZy = sZy[number]
        sZx = sZx[number]
        tri = Delaunay(points)
        e = ax.triplot(points[:,0], points[:,1], tri.simplices, color="#888", linewidth=0.5)
        elems.append(e)
        e = ax.scatter(points[:,0], points[:,1], color=vcolorify((sZx**2+sZy**2)**0.5))
        elems.append(e)
        for df in dfs:
            datestr = dfs[df].iloc[h]["date"]
            break
        e = ax.set_title(datestr)
        elems.append(e)
        elems += perimeters(ax, shape)

        return elems
    else:

        # 1. グリッド点の処理

        interp, sX, sY, sZ = gridfunc(target, h, dfs, stations)
        Z = interp(gX,gY)
        # 内挿できなかった格子点を除外する。
        X = gX[~np.isnan(Z)]
        Y = gY[~np.isnan(Z)]
        Z = Z[~np.isnan(Z)]

        elems = []

        colors = vcolorify(Z)

        e = ax.scatter(X, Y, c=colors)

        # 2. 測定局の処理

        # Compute Delaunay
        points = np.array([sX,sY]).T
        # 欠測を除く。
        number = ~np.isnan(sZ)
        points = points[number]
        sZ = sZ[number]
        tri = Delaunay(points)
        e = ax.triplot(points[:,0], points[:,1], tri.simplices, color="#888", linewidth=0.5)
        elems.append(e)
        e = ax.scatter(points[:,0], points[:,1], color=vcolorify(sZ), ec="#000")
        elems.append(e)
        e = ax.set_title(datestr)
        elems.append(e)
        elems += perimeters(ax, shape)

        return elems


#constants
#J2K = 0.120273 # K/J 20110521 modified.
MassMe = 16e-3 # kg/mol

#Physical propertires
kB   = 1.380662e-23 # J/K
NA   = 6.022045e23  # 1
J2K =  1.0/(kB*NA) # K/J
NkB  = kB * NA * 1e-3 # kJ/mol/K
h    = 6.626176e-34 # J s Planck's h
cc   = 2.99792e10

# use ABC values in NIST database
# http://webbook.nist.gov/cgi/cbook.cgi?ID=C287923&Mask=4&Type=ANTOINE&Plot=on
def VaporPressure(T, A=5.40221, B=1838.675, C=-31.737):  # in Pa
    return 10.0**(A - B / (C + T)) * 101325.0


@lru_cache
def load(fn, drop=[]):
    try:
        df = pd.read_csv(fn)
        if "TEMP" in df.columns:
            df["TEMP"] = df["TEMP"] / 10
        df["date"] = pd.to_datetime(df["date"])
        # 日付けと、重複して積算している量(合計量)を排除する。
        df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

        if "HUM" in df.columns:
            df["Water"] = np.log(VaporPressure(df["TEMP"]+273.15)*df["HUM"]/100)
            df = df.drop(columns=["HUM"])

        for col in drop:
            if col in df.columns:
                df = df.drop(columns=col)

        return df
    except:
        return None


@lru_cache(maxsize=5000)
def retrieve(location, date, hours=24, drop=tuple(), PATH=""):
    """とりあえず、てもとのディスク上にあるデータを返すのみ。

    Args:
        location: location id (tenbou data)
        date (_type_): year, month, day, and hour. (minutes and seconds are ignored)
        span (int, optional): time span. 24 means that the oldest data should be 24 hours before the date. Defaults to 24.
        drop (list of strings): あらかじめ削除しておくカラム
    """
    # 日付はここで使う。
    drop0 = tuple([x for x in drop if x != "date"])

    date = pd.Timestamp(date)
    df = None
    year = date.year
    head = pd.Timestamp(date - datetime.timedelta(hours=hours))
    pref = location // 1000000
    while True:
        fn=f"{PATH}{pref}/{year}/j{pref}{year}_{location}.csv"
        # assert os.path.exists(fn), f"{fn} not found."
        df1 = load(fn=fn, drop=drop0)
        if df1 is None:
            break
        # 年がかわると、測定項目が変わる可能性がある。
        # その場合は、項目が少ないほうにあわせる。
        year -= 1
        if df is None:
            df = df1
        else:
            df = pd.concat((df, df1), ignore_index=True)
        if df1 is not None and df1.iloc[0].loc["date"] < head:
            break
    if df is not None:
        df = df[df["date"]>=head]
        df = df[df["date"]<=date]

        # 日付をとりのぞく。
        if "date" in drop:
            df = df.drop(columns="date")
            return df

        return df.sort_values(by="date")
    return df
