/*-------------------------------------------------------  학습데이터 */
%LET TRAIN_RAW=  SUSER40.AMTRAW_LINE39_A1_20210401;  /*학습데이터 AMT */
%LET YYYYMM ='202104' ; /* EVENT월 */
%LET LINE=39;  /*생성할 중분류코드 */
%LET BIZ=A1;  /*생성할 유형*/
%LET END_DATE1 = '20210415';
%LET NUM=20210416;  /*TA값 시점 NUM*/
%LET TRAINNUM=20210401;  /*학습데이터번호 */

PROC SQL;
CREATE TABLE GOOD_DM AS 
SELECT BD_ITEM_LCLS_CD AS 대분류코드
          ,BD_ITEM_LCLS_NM AS 대분류명
          ,BD_ITEM_MCLS_CD AS LINE_CD
          ,BD_ITEM_MCLS_NM AS LINE_NM
          ,BD_ITEM_SCLS_CD AS CLASS_CD
		  ,BD_ITEM_SCLS_NM AS CLASS_NM
		  ,BD_ITEM_DCLS_CD AS 세분류코드
		  ,BD_ITEM_DCLS_NM AS 세분류명
          ,COMPRESS(BD_ITEM_CD) AS GOODS_CD
          ,COMPRESS(BD_ITEM_NM) AS GOODS_NM
  FROM GSSELDW.TB_DM_BD_ITEM_BASE
  WHERE BIZU_CD='1' /*편의점코드*/
;
QUIT;
PROC SORT DATA=TEST2.TS_MS_BIZPL OUT=공결점포01 ; BY ORIGIN_BIZPL_CD  DESCENDING OPEN_DT; RUN;
PROC SORT DATA=공결점포01 OUT=공결점포02 DUPOUT=공결점포03 NODUPKEY; BY ORIGIN_BIZPL_CD; RUN;

PROC SORT DATA=TEST2.TS_MS_BIZPL OUT=공01 ; BY ORIGIN_BIZPL_CD  OPEN_DT; RUN;
PROC SORT DATA=공01 OUT=공02 DUPOUT=공03 NODUPKEY; BY ORIGIN_BIZPL_CD; RUN;

PROC SORT DATA=TEST2.TS_MS_BIZPL OUT=공04 ; BY ORIGIN_BIZPL_CD   DESCENDING OPEN_DT; RUN;
PROC SORT DATA=공04 OUT=공05 DUPOUT=공06 NODUPKEY; BY ORIGIN_BIZPL_CD; RUN;

PROC SQL; 
CREATE TABLE  공07 AS
SELECT A.ORIGIN_BIZPL_CD,A.BIZPL_NM,A.OPEN_DT ,B.CLOSE_DT 
FROM 공02 AS A
LEFT JOIN  공05 AS B
ON A.ORIGIN_BIZPL_CD=B.ORIGIN_BIZPL_CD
;QUIT; 

DATA AMT_LINE_&BIZ. ;
SET &TRAIN_RAW. ;
RUN;

/*이벤트 테이블*/
 DATA EVENT;
SET     ZTC_LIB.LAST_EVENT;
WHERE YYYYMM= &YYYYMM. ;
       RUN;

PROC SQL;
CREATE TABLE AMT_LINE2 AS
SELECT A.BIZPL_CL_DIV_CD
            ,A.ORIGIN_BIZPL_CD
            ,B.GOODS_REGION_CD
            ,A.KEY
            ,A.LINE_CD
            ,D.CLASS_CD
			,D.CLASS_NM
            ,A.GOODS_CD AS GOODS_CD_NOW
            ,D.GOODS_NM AS GOODS_NM_NOW
            ,CASE WHEN C.SET_COMPOSIT_GOODS_CD ^='' THEN C.SET_COMPOSIT_GOODS_CD ELSE A.GOODS_CD END AS GOODS_CD
            ,CASE WHEN E.GOODS_NM^='' THEN E.GOODS_NM ELSE D.GOODS_NM END  AS GOODS_NM
            ,A.SALE_DAY
            ,CASE WHEN C.SET_COMPOSIT_GOODS_CD^='' THEN A.QTY*C.CNVS_QTY ELSE A.QTY END AS QTY
			,A.AMT 
            ,CALCULATED QTY / SALE_DAY AS DQTY
			,A.AMT/SALE_DAY AS DAMT
FROM AMT_LINE_&BIZ. AS A
LEFT JOIN 공결점포02 AS B 
ON A.ORIGIN_BIZPL_CD = B.ORIGIN_BIZPL_CD
LEFT JOIN (SELECT * FROM GSSELDW.TH_MS_GOODS_COMPOSIT_CS WHERE CNVS_QTY>=1)  AS C 
ON A.GOODS_CD = C.GOODS_CD AND B.GOODS_REGION_CD = C.GOODS_REGION_CD 
LEFT JOIN GOOD_DM AS D
ON A.GOODS_CD = D.GOODS_CD
LEFT JOIN GOOD_DM AS E 
ON C.SET_COMPOSIT_GOODS_CD = E.GOODS_CD
;
QUIT;


PROC SQL; 
CREATE TABLE AMT_LINE3 AS
SELECT A.*,
CASE WHEN C.'2+1'N >0 THEN '2+1' 
WHEN C.'1+1'N >0 THEN '1+1' 
WHEN C.'할인'N>0 THEN '할인' 
ELSE '없음'
END AS EVENT 
FROM AMT_LINE2 AS A
LEFT JOIN EVENT AS C
ON A.GOODS_CD_NOW=C.PRSNT_GOODS_CD 
;
QUIT; 

/*-------------------------------------------------------- 단품별 이상치 제거*/
PROC SORT DATA=AMT_LINE3; BY ORIGIN_BIZPL_cD CLASS_CD EVENT ; QUIT; 
PROC MEANS DATA=AMT_LINE3 MEAN Q3 Q1 NOPRINT;
BY ORIGIN_BIZPL_CD CLASS_CD EVENT ;
VAR QTY;
OUTPUT OUT=QTY_SIX
STDDEV=STDD  
Q3 = SIX_Q3
Q1 = SIX_Q1
MEAN = SIX_MEAN;
RUN;
PROC MEANS DATA=AMT_LINE3 MEAN Q3 Q1 NOPRINT;
BY ORIGIN_BIZPL_CD CLASS_CD EVENT ;
VAR AMT;
OUTPUT OUT=AMT_SIX
STDDEV=STDD  
Q3 = SIX_Q3
Q1 = SIX_Q1
MEAN = SIX_MEAN;
RUN;
DATA QTY_SIX01;
SET QTY_SIX;
IQR = SIX_Q3 - SIX_Q1;
/*LP = SIX_Q1-1.5*(SIX_Q3-SIX_Q1);*/
/*UP = SIX_Q3+1.5*(SIX_Q3-SIX_Q1);*/
LP = SIX_MEAN-2*STDD;
UP = SIX_MEAN+2*STDD;

RUN;
DATA AMT_SIX01;
SET AMT_SIX;
IQR = SIX_Q3 - SIX_Q1;
/*LP = SIX_Q1-1.5*(SIX_Q3-SIX_Q1);*/
/*UP = SIX_Q3+1.5*(SIX_Q3-SIX_Q1);*/
UP = SIX_MEAN+2*STDD;
LP = SIX_MEAN-2*STDD;
RUN;

PROC SQL;
CREATE TABLE AMT_LINE02 AS
SELECT A.*
			,B.UP AS QTY_UP
			,C.UP AS AMT_UP
			,CASE WHEN (A.QTY > B.UP OR A.QTY< B.LP) THEN 1 ELSE 0 END AS DEL1
			,CASE WHEN (A.AMT > C.UP OR A.AMT<C.LP) THEN 1 ELSE 0 END AS DEL2
FROM AMT_LINE3 AS A
LEFT JOIN QTY_SIX01 AS B
ON A.ORIGIN_BIZPL_CD = B.ORIGIN_BIZPL_CD AND A.CLASS_CD = B.CLASS_CD AND A.EVENT=B.EVENT
LEFT JOIN AMT_SIX01 AS C 
ON A.ORIGIN_BIZPL_CD = C.ORIGIN_BIZPL_CD AND A.CLASS_CD = C.CLASS_CD AND A.EVENT=C.EVENT
HAVING (DEL1=0 AND DEL2=0) 
;
QUIT;

PROC SQL; 
CREATE TABLE AMT_LINE03 AS
SELECT BIZPL_CL_DIV_CD, ORIGIN_BIZPL_CD, SALE_DAY,LINE_CD,GOODS_CD,GOODS_NM
,SUM(QTY) AS QTY
,SUM(AMT) AS AMT
,CALCULATED QTY /SALE_DAY AS DQTY
,CALCULATED AMT /SALE_DAY AS DAMT 
FROM AMT_LINE02 
GROUP BY 1,2,3,4,5,6
;
QUIT; 


/*---------------------------------------------------TFIDF구하기 */
/*TF*/
PROC SQL;
CREATE TABLE BIZPL_TF AS
SELECT A.BIZPL_CL_DIV_CD
			,A.ORIGIN_BIZPL_CD 
			,A.LINE_CD
			,A.GOODS_CD
			,A.DQTY 
			,B.MAX
			,(0.1+0.9*(A.DQTY / B.MAX)) AS TF
FROM AMT_LINE03 AS A
LEFT JOIN (SELECT ORIGIN_BIZPL_CD
							,LINE_CD
							,MAX(DQTY) AS MAX
				FROM AMT_LINE03
				GROUP BY ORIGIN_BIZPL_CD
								,LINE_CD) AS B 
ON A.ORIGIN_BIZPL_CD = B.ORIGIN_BIZPL_CD AND A.LINE_CD = B.LINE_CD
;
QUIT;

/*IDF*/
PROC SQL;
CREATE TABLE BIZPL_IDF AS
SELECT A.BIZPL_CL_DIV_CD
			,A.LINE_CD
			,A.GOODS_CD
			,B.TOTAL
			,COUNT(DISTINCT A.ORIGIN_BIZPL_CD) AS STORE
			,LOG(B.TOTAL/(CALCULATED STORE +1)) +0.1 AS IDF
			,CALCULATED STORE / B.TOTAL AS RATIO
FROM AMT_LINE03 AS A/*판매점포수*/
LEFT JOIN (SELECT BIZPL_CL_DIV_CD, COUNT(DISTINCT ORIGIN_BIZPL_CD) AS TOTAL FROM AMT_LINE03 WHERE DQTY^=0
				GROUP BY BIZPL_CL_DIV_CD) AS B ON A.BIZPL_CL_DIV_CD = B.BIZPL_CL_DIV_CD/*전체점포수*/
WHERE A.DQTY^=0
GROUP BY A.BIZPL_CL_DIV_CD
			,A.LINE_CD
			,A.GOODS_CD
			,B.TOTAL
;
QUIT;

PROC MEANS DATA=BIZPL_IDF MEAN Q3 Q1 NOPRINT;
BY BIZPL_CL_DIV_CD LINE_CD;
VAR IDF;
OUTPUT OUT=IDF_SIX
Q3 = SIX_Q3
Q1 = SIX_Q1
MEAN = SIX_MEAN;
RUN;

DATA IDF_SIX01;
SET IDF_SIX;
IQR = SIX_Q3 - SIX_Q1;
LP = SIX_Q1-1.5*(SIX_Q3-SIX_Q1);
UP = SIX_Q3+1.5*(SIX_Q3-SIX_Q1);
RUN;

PROC SQL;
CREATE TABLE BIZPL_IDF01 AS
SELECT A.*
			,B.UP
			,CASE WHEN A.IDF > B.UP THEN 1 ELSE 0 END AS DEL
FROM BIZPL_IDF AS A
LEFT JOIN IDF_SIX01 AS B ON A.BIZPL_CL_DIV_CD = B.BIZPL_CL_DIV_CD AND A.LINE_CD = B.LINE_CD
HAVING DEL=0
;
QUIT;

/*TFIDF*/
PROC SQL;
CREATE TABLE BIZPL_TFIDF_&BIZ. AS
SELECT A.BIZPL_CL_DIV_CD
			,A.ORIGIN_BIZPL_CD
			,C.LINE_CD
			,C.LINE_NM
			,C.CLASS_CD
			,C.CLASS_NM
			,A.GOODS_CD
			,C.GOODS_NM
			,A.DQTY
			,A.TF
			,B.IDF
			,A.MAX
			,TF * IDF AS TFIDF
FROM BIZPL_TF AS A
INNER JOIN BIZPL_IDF01 AS B ON A.GOODS_CD = B.GOODS_CD AND A.BIZPL_CL_DIV_CD = B.BIZPL_CL_DIV_CD
LEFT JOIN (SELECT DISTINCT LINE_CD, LINE_NM, CLASS_CD, CLASS_NM, GOODS_CD, GOODS_NM FROM GOOD_DM) AS C ON A.GOODS_CD = C.GOODS_CD
HAVING GOODS_NM^='' AND TFIDF>0
ORDER BY BIZPL_CL_DIV_CD, LINE_CD, ORIGIN_BIZPL_CD, TFIDF DESC
;
QUIT;

DATA BIZPL_TFIDF_01_&BIZ.;
SET BIZPL_TFIDF_&BIZ.;
IF INDEX(CLASS_NM,'온라인')>0 THEN DEL=1;ELSE DEL=0;
IF DEL=0;
RUN;

PROC SQL;
CREATE TABLE GOODS_SEQ AS
SELECT DISTINCT A.BIZPL_CL_DIV_CD, A.LINE_CD, A.GOODS_CD
FROM BIZPL_TFIDF_01_&BIZ. AS A
ORDER BY LINE_CD, BIZPL_CL_DIV_CD, GOODS_CD
;
QUIT;

DATA GOODS_SEQ01;
SET GOODS_SEQ;
BY LINE_CD BIZPL_CL_DIV_CD;
N+1-FIRST.BIZPL_CL_DIV_CD*N;
RUN;

PROC SQL;
CREATE TABLE ORIGIN_SEQ AS
SELECT DISTINCT A.LINE_CD, A.BIZPL_CL_DIV_CD, A.ORIGIN_BIZPL_CD, B.BIZPL_NM
FROM BIZPL_TFIDF_01_&BIZ. AS A 
LEFT JOIN 공결점포02 AS B ON A.ORIGIN_BIZPL_CD = B.ORIGIN_BIZPL_CD
ORDER BY LINE_CD, BIZPL_CL_DIV_CD, ORIGIN_BIZPL_CD
;
QUIT;

DATA ORIGIN_SEQ01;
SET ORIGIN_SEQ;
BY LINE_CD BIZPL_CL_DIV_CD;
N1+1-FIRST.BIZPL_CL_DIV_CD*N1;
KEY = COMPRESS("A_"||N1);
RUN;
	PROC SQL NOPRINT;
	SELECT COMPRESS(PUT(MAX(N),VARCHAR3.))	INTO : GOODS_N	FROM GOODS_SEQ01 WHERE (LINE_CD = "&LINE" AND BIZPL_CL_DIV_CD = "&BIZ") ;
	SELECT COMPRESS("A_"||(PUT(MAX(N1),VARCHAR4.)))	INTO : ORIGIN_N	FROM ORIGIN_SEQ01 WHERE (LINE_CD = "&LINE" AND BIZPL_CL_DIV_CD = "&BIZ") ;
	QUIT ; 

	PROC SQL NOPRINT;
	CREATE TABLE ORIGIN_MAX AS
	SELECT DISTINCT BIZPL_CL_DIV_CD, ORIGIN_BIZPL_CD
				,LINE_CD
				,MAX
	FROM BIZPL_TFIDF_01_&BIZ.
	WHERE LINE_CD = "&LINE" AND BIZPL_CL_DIV_CD = "&BIZ"
	;
	QUIT;
		
	PROC SQL NOPRINT;
	CREATE TABLE BIZPL_TFIDF01 AS
	SELECT A.BIZPL_CL_DIV_CD
				,A.ORIGIN_BIZPL_CD
				,A.LINE_CD
				,A.GOODS_CD
				,A.GOODS_NM
				,A.TFIDF
				,B.N
				,C.KEY
	FROM BIZPL_TFIDF_01_&BIZ. AS A
	INNER JOIN GOODS_SEQ01 AS B ON A.BIZPL_CL_DIV_CD = B.BIZPL_CL_DIV_CD AND A.LINE_CD = B.LINE_CD AND A.GOODS_CD = B.GOODS_CD AND B.LINE_CD = "&LINE"
	INNER JOIN ORIGIN_SEQ01 AS C ON A.BIZPL_CL_DIV_CD = C.BIZPL_CL_DIV_CD AND A.ORIGIN_BIZPL_CD = C.ORIGIN_BIZPL_CD AND A.LINE_CD = C.LINE_CD AND C.BIZPL_CL_DIV_CD = "&BIZ"
	ORDER BY BIZPL_CL_DIV_CD, KEY, N
	;
	QUIT;

	PROC TRANSPOSE DATA=BIZPL_TFIDF01 OUT=BIZPL_TFIDF02;
	BY BIZPL_CL_DIV_CD KEY;
	ID N;
	VAR TFIDF;
	RUN;

	/*코사인유사도*/
	PROC DISTANCE DATA=BIZPL_TFIDF02 OUT=COS METHOD=COSINE SHAPE=SQUARE;
	BY BIZPL_CL_DIV_CD;
	VAR RATIO('1'N-"&GOODS_N"N);
	ID KEY;
	RUN;

	PROC TRANSPOSE DATA=COS OUT=COS01_&BIZ.;
	BY KEY;
	VAR A_1 - &ORIGIN_N;
	RUN;

	/*유사점*/
    PROC SQL NOPRINT;
    CREATE TABLE COS02 AS
    SELECT A.*
                ,CASE WHEN 1>=A.COL1>=0.9 THEN 1 ELSE 0 END AS SEG
    FROM COS01_&BIZ. AS A
    WHERE _NAME_^=KEY
    HAVING SEG=1
    ORDER BY A.KEY, A.COL1 DESC
    ;
    QUIT;

/*---------------------------------------------------------------------------유사도별 */
	%MACRO TEMP ();
	%DO J=90 %TO 98  ; 
	%PUT -------------------J=&J.;
/*	%LET J=81;*/
	DATA COS02_2 ;
	SET COS02 ;
	X=COL1*100;
	XX=&J.;
	IF X>=&J. THEN T=1 ;
IF	T=1;
	DROP X XX T ;
RUN;
	PROC SORT DATA=COS02_2 ;BY KEY DESCENDING COL1;QUIT; 
	PROC SQL NOPRINT;
	CREATE TABLE COS03_&J AS
	SELECT A.*
			,C.ORIGIN_BIZPL_CD AS BF_ORIGIN
			,C.BIZPL_NM AS BF_BIZ
			,E.MAX AS BF_MAX
			,D.ORIGIN_BIZPL_CD AS AF_ORIGIN
			,D.BIZPL_NM AS AF_BIZ
			,"&LINE" AS LINE_CD
	FROM COS02_2 AS A
	LEFT JOIN ORIGIN_SEQ01 AS C ON A.KEY =C.KEY AND C.LINE_CD = "&LINE" AND C.BIZPL_CL_DIV_CD = "&BIZ"
	LEFT JOIN ORIGIN_SEQ01 AS D ON A._NAME_ = D.KEY AND D.LINE_CD = "&LINE" AND D.BIZPL_CL_DIV_CD = "&BIZ"
	LEFT JOIN ORIGIN_MAX AS E ON C.ORIGIN_BIZPL_CD = E.ORIGIN_BIZPL_CD
	;
	QUIT;

/*	NOTE: Table WORK.ORIGIN_MAX created, with 2080 rows and 4 columns.*/


    PROC SQL;
    CREATE TABLE ITEM_Y_BASED AS
    SELECT "&BIZ" AS BIZPL_CL_DIV_CD
                ,A.BF_ORIGIN
                ,A.BF_BIZ
                ,A.BF_MAX
                ,B.GOODS_CD
                ,SUM(B.TFIDF*A.COL1) AS CUM_TF_COL
                ,SUM(A.COL1) AS CUM_COL1
                ,CALCULATED CUM_TF_COL / CALCULATED CUM_COL1 AS PRE_TFIDF
    FROM COS03_&J AS A
    INNER JOIN BIZPL_TFIDF_01_&BIZ. AS B ON A.AF_ORIGIN = B.ORIGIN_BIZPL_CD AND B.LINE_CD = "&LINE" AND B.BIZPL_CL_DIV_CD = "&BIZ"
    GROUP BY "&BIZ"
                ,A.BF_ORIGIN
                ,A.BF_BIZ
                ,A.BF_MAX
                ,B.GOODS_CD
    ;
    QUIT;

 

    PROC SQL;
    CREATE TABLE ITEM_Y_BASED01 AS 
    SELECT A.*, B.IDF, (((PRE_TFIDF / IDF)-0.1)/0.9)*BF_MAX AS PRE_QTY
    FROM ITEM_Y_BASED AS A
    LEFT JOIN BIZPL_IDF01 AS B ON A.GOODS_CD = B.GOODS_CD
    ;
    QUIT;

 

    PROC SQL;
    CREATE TABLE ITEM_BASED_&J AS
    SELECT A.BIZPL_CL_DIV_CD
                ,A.BF_ORIGIN
                ,A.BF_BIZ
                ,B.LINE_CD
                ,B.LINE_NM
                ,B.CLASS_CD
                ,B.CLASS_NM
                ,A.GOODS_CD
                ,B.GOODS_NM
                ,A.PRE_TFIDF
                ,A.PRE_QTY
                ,CASE WHEN C.GOODS_CD^='' THEN 1 ELSE 0 END AS NOW_YN
                ,C.TFIDF AS REAL_TFIDF
                ,C.DQTY AS REAL_QTY
    FROM ITEM_Y_BASED01 AS A
    LEFT JOIN GOOD_DM AS B ON A.GOODS_CD = B.GOODS_CD
    LEFT JOIN BIZPL_TFIDF_01_&BIZ. AS C ON A.BF_ORIGIN = C.ORIGIN_BIZPL_CD AND A.GOODS_CD = C.GOODS_CD AND C.BIZPL_CL_DIV_CD = "&BIZ" AND C.LINE_CD = "&LINE"
    ORDER BY BF_ORIGIN, PRE_QTY DESC
    ;
    QUIT;




		%END ;
		%MEND ;

		%TEMP();


		
/*----------------------------------------------------------------------------- 오차구하기*/
DATA RMSE_&BIZ.;
RUN; 


%MACRO RMSE();

%DO J=90 %TO 98 ;

DATA ITEM_BASET_&J.;
SET ITEM_BASED_&J ;
IF NOW_YN=1 ;
RUN;

DATA DIFF_&J. ;
SET  ITEM_BASET_&J.;
TFIDF_DIFF=PRE_TFIDF-REAL_TFIDF ;
QTY_DIFF =PRE_QTY-REAL_QTY ;
TFIDF_SSE=TFIDF_DIFF*TFIDF_DIFF ;
QTY_SSE=QTY_DIFF*QTY_DIFF ;
SIM = &J. ;
RUN;

PROC SQL; 
CREATE TABLE RMSE_&J. AS
SELECT &J. AS SIM,SQRT(TFIDF_MSE) AS TFIDF_RMSE, SQRT(QTY_MSE) AS QTY_RMSE 
FROM (SELECT &J. AS SIM, MEAN(TFIDF_SSE) AS TFIDF_MSE ,MEAN(QTY_SSE) AS QTY_MSE
			FROM DIFF_&J.
			GROUP BY 1)
;
QUIT; 
DATA RMSE_&BIZ. ;
SET RMSE_&J. RMSE_&BIZ. ;
RUN;

PROC SORT DATA= RMSE_&BIZ. ; BY SIM ;QUIT; 

%END ;
%MEND ;


%RMSE();


	PROC SQL;
	CREATE TABLE DIFF_RESULT AS
	SELECT A.*, B.QTY_RMSE  AS DIFF_T, (1-B.QTY_RMSE /A.QTY_RMSE ) AS BETA
, CASE WHEN CALCULATED BETA<0 THEN 'A' ELSE 'B' END AS DEL
	FROM RMSE_&BIZ. AS A
	LEFT JOIN RMSE_&BIZ. AS B ON A.SIM = B.SIM+1
	WHERE A.SIM ^=.
	;
	QUIT;

	PROC SQL NOPRINT;
	SELECT COUNT(DISTINCT SIM)  INTO : CNT	FROM DIFF_RESULT WHERE DEL='B';
	QUIT ; 

	%IF &CNT>0 %THEN %DO; 

	PROC SQL;
	CREATE TABLE DIFF_RESULT01 AS
	SELECT A.*
	FROM DIFF_RESULT AS A
	INNER JOIN (SELECT MIN(SIM) AS SIM FROM DIFF_RESULT WHERE BETA>0) AS B ON A.SIM = B.SIM-1 
	;
	QUIT;
	%END;
	%IF &CNT=0 %THEN %DO;
	PROC SQL;
	CREATE TABLE DIFF_RESULT01 AS
	SELECT A.*
	FROM DIFF_RESULT AS A
	INNER JOIN (SELECT MAX(SIM) AS SIM FROM DIFF_RESULT WHERE BETA<0) AS B ON A.SIM = B.SIM
	;
	QUIT;
	%END;

PROC SQL NOPRINT ;
SELECT COMPRESS(PUT(SIM,2.)) INTO: COS FROM DIFF_RESULT01 ;QUIT; 

PROC SQL; 
CREATE TABLE ORIGIN_&BIZ. AS
SELECT  BF_ORIGIN,COUNT(DISTINCT AF_ORIGIN) AS N
FROM COS03_&COS 
GROUP BY 1
HAVING N>=5
;
QUIT; 
PROC SQL; 
CREATE TABLE TRAIN_SET_&BIZ. AS
SELECT A.*
FROM ITEM_BASED_&COS AS A
INNER JOIN ORIGIN_&BIZ. AS B
ON A.BF_ORIGIN=B.BF_ORIGIN ;
QUIT; 

/*유사점리스트*/
PROC SQL; 
CREATE TABLE SUSER40.ORIGIN_COS&TRAINNUM AS
SELECT  BF_ORIGIN,BF_BIZ, AF_ORIGIN
FROM COS03_&COS 
WHERE BF_ORIGIN  IN (SELECT DISTINCT  BF_ORIGIN FROM ORIGIN_&BIZ.)

;
QUIT; 
DATA ZTC_LIB.ORIGIN_A1;
SET SUSER40.ORIGIN_COS20210401 ;
RUN;
/*점포별 중분류 상위 80%구하기 */

DATA 학습Y_AMT ;
SET SUSER40.&BIZ._&line.예측Y_RAW&NUM.;
RUN;

PROC SQL; 
CREATE TABLE ORIGIN_LINE_N AS
SELECT BIZPL_CL_DIV_CD, ORIGIN_BIZPL_CD, LINE_CD,SUM(DQTY) AS SUM_DQTY
FROM SUSER40.&BIZ._&line.예측Y_RAW20210416
/*학습Y_AMT*/
GROUP BY 1,2 ,3
;
QUIT; 
PROC SQL; 
CREATE TABLE LINE80 AS
SELECT A.*,B.SUM_DQTY, DQTY/SUM_DQTY AS QTY_RATIO
FROM  학습Y_AMT AS A
LEFT JOIN ORIGIN_LINE_N AS B
ON A.ORIGIN_BIZPL_CD=B.ORIGIN_BIZPL_CD
ORDER BY A.BIZPL_CL_DIV_CD, A.ORIGIN_BIZPL_CD, DQTY DESC 
;
QUIT; 

DATA LINE80_2 ;
SET LINE80;
BY BIZPL_CL_DIV_CD ORIGIN_BIZPL_CD ;
IF FIRST.ORIGIN_BIZPL_CD THEN SUM_RATIO=QTY_RATIO ;
ELSE SUM_RATIO+QTY_RATIO ;
IF SUM_RATIO<=0.8 THEN 상위Y=1 ;
ELSE 상위Y=0 ;
RUN;


/*유사점 취급율 붙이기*/
PROC SQL; 
CREATE TABLE AMT_FIN AS
SELECT ORIGIN_BIZPL_CD, GOODS_CD,SUM(QTY) AS QTY
FROM AMT_LINE2 
GROUP BY 1,2
;
QUIT; 

PROC SQL; 
CREATE TABLE SIM_&BIZ._1 AS
SELECT A.*,C.유사점수,B.AF_ORIGIN 
FROM TRAIN_SET_&BIZ. AS A
LEFT JOIN COS03_&COS. AS B
ON A.BF_ORIGIN=B.BF_ORIGIN 
LEFT JOIN (SELECT BF_ORIGIN, COUNT(AF_ORIGIN) AS 유사점수 
				FROM COS03_&COS. 
				GROUP BY 1) AS C
ON A.BF_ORIGIN =C.BF_ORIGIN 
;
QUIT; 


PROC SQL; 
CREATE TABLE SIM_&BIZ._2 AS 
SELECT A.BF_ORIGIN, A.GOODS_CD,A.유사점수 
,SUM(CASE WHEN B.QTY ^=. THEN 1 ELSE 0 END)  AS 유사점취급 
FROM SIM_&BIZ._1 AS A
LEFT JOIN AMT_FIN AS B
ON A.AF_ORIGIN =B.ORIGIN_BIZPL_CD AND A.GOODS_CD=B.GOODS_CD
GROUP BY 1,2,3
;
QUIT; 
PROC SQL; 
CREATE TABLE TRAIN_SET_&BIZ._P AS
SELECT A.*,D.상위Y AS TA,B.유사점취급/유사점수 AS 유사점취급율 ,C.RATIO AS 유형취급율 
FROM TRAIN_SET_&BIZ. AS A
LEFT JOIN SIM_&BIZ._2 AS B
ON A.BF_ORIGIN=B.BF_ORIGIN AND A.GOODS_CD =B.GOODS_CD
LEFT JOIN ZTC_LIB.BIZPL_RATIO AS C
ON A.GOODS_CD=C.GOODS_CD AND A.BIZPL_CL_DIV_CD=C.BIZPL_CL_DIV_CD
INNER JOIN LINE80_2 AS D
ON A.BF_ORIGIN=D.ORIGIN_BIZPL_CD AND A.GOODS_CD=D.GOODS_CD
INNER JOIN (SELECT ORIGIN_BIZPL_CD, CLOSE_DT FROM  공07
					WHERE CLOSE_DT > &END_DATE1. OR CLOSE_DT ='' )AS E
ON A.BF_ORIGIN=E.ORIGIN_BIZPL_CD
WHERE C.SEG='BF' 
ORDER BY A.BIZPL_CL_DIV_CD, A.BF_ORIGIN,A.LINE_NM,A.CLASS_NM
;
QUIT; 


PROC SQL; 
CREATE TABLE SUSER40.TRAIN_&BIZ._&line._80_TRAIN&TRAINNUM. AS
SELECT A.* ,B.상위Y AS TA
FROM TRAIN_SET_&BIZ. AS A
INNER JOIN LINE80_2 AS B
ON A.BF_ORIGIN=B.ORIGIN_BIZPL_CD AND A.GOODS_CD=B.GOODS_CD
INNER JOIN (SELECT ORIGIN_BIZPL_CD, CLOSE_DT FROM  공07
					WHERE CLOSE_DT > &END_DATE1. OR CLOSE_DT ='' )AS C
ON A.BF_ORIGIN=C.ORIGIN_BIZPL_CD
;
QUIT;




/* 정확도 계산용*/
PROC SQL; 
CREATE TABLE DEL_&BIZ. AS
SELECT BF_ORIGIN ,SUM(TA) AS TASUM ,COUNT(*) AS N
FROM  SUSER40.TRAIN_&BIZ._&line._80_TRAIN&TRAINNUM.
GROUP BY 1
HAVING TASUM=0 OR (TASUM=N)

;
QUIT; 


PROC SQL; 
CREATE TABLE TRAIN_&BIZ. AS
SELECT A.*,B.DQTY AS TA_QTY 
FROM  SUSER40.TRAIN_&BIZ._&line._80_TRAIN&TRAINNUM. AS A
LEFT JOIN LINE80_2  AS B
ON A.BF_ORIGIN=B.ORIGIN_BIZPL_CD AND A.GOODS_CD=B.GOODS_CD
WHERE BF_ORIGIN NOT IN (SELECT BF_ORIGIN FROM DEL_&BIZ.) /*TA가 모두 0이거나 1인 점포제외*/
ORDER BY BF_ORIGIN;
QUIT; 

data test2.danbi_TRAIN_line&LINE._20210316;
set  TRAIN_&BIZ. ;
run;/*이후 파이썬 작업*/

/**/
/*PROC SQL; */
/*CREATE TABLE SUSER40.TRAIN_&BIZ._80_&NUM AS*/
/*SELECT A.* ,B.상위Y AS TA*/
/*FROM TRAIN_SET_&BIZ. AS A*/
/*INNER JOIN LINE80_2 AS B*/
/*ON A.BF_ORIGIN=B.ORIGIN_BIZPL_CD AND A.GOODS_CD=B.GOODS_CD*/
/*INNER JOIN (SELECT ORIGIN_BIZPL_CD, CLOSE_DT FROM  공07*/
/*					WHERE CLOSE_DT > &END_DATE1. OR CLOSE_DT ='' )AS C*/
/*ON A.BF_ORIGIN=C.ORIGIN_BIZPL_CD*/
/*;*/
/*QUIT; */
