(() => {
  const e = React.createElement;
  const { useEffect, useMemo, useRef, useState } = React;
  const pages = [
    ["overview", "總覽", "01"], ["holdings", "即時持股", "02"],
    ["performance", "績效歷史", "03"], ["history", "持股換倉", "04"],
    ["risk", "風險系統", "05"]
  ];
  const money = v => v == null ? "—" : new Intl.NumberFormat("zh-TW", {style:"currency",currency:"TWD",maximumFractionDigits:0}).format(v);
  const num = (v,d=2) => v == null ? "—" : Number(v).toLocaleString("zh-TW",{minimumFractionDigits:d,maximumFractionDigits:d});
  const pct = (v,d=2) => v == null ? "—" : `${(v*100).toFixed(d)}%`;
  const yi = (v,d=2) => v == null ? "—" : `${num(v/1e8,d)} 億`;
  const cls = v => v > 0 ? "positive" : v < 0 ? "negative" : "";
  const api = async path => { const r=await fetch(path); if(!r.ok) throw new Error(`${r.status} ${path}`); return r.json(); };

  function Chart({option, small=false}) {
    const ref=useRef(null),chart=useRef(null);
    useEffect(()=>{ if(!ref.current || !window.echarts)return; chart.current=echarts.init(ref.current); const resize=()=>chart.current?.resize(); window.addEventListener("resize",resize); return()=>{window.removeEventListener("resize",resize);chart.current?.dispose();chart.current=null}; },[]);
    useEffect(()=>{ chart.current?.setOption(option,{notMerge:false,lazyUpdate:true}); },[option]);
    return e("div",{ref,className:`chart${small?" small":""}`});
  }
  function Card({title,children,note,className=""}) { return e("section",{className:`card section ${className}`},e("div",{className:"section-head"},e("div",{className:"section-title"},title),note&&e("div",{className:"section-note"},note)),children); }
  function Kpi({label,value,note,tone=""}) { return e("div",{className:"card kpi"},e("div",{className:"kpi-label"},label),e("div",{className:`kpi-value ${tone}`},value),e("div",{className:"kpi-note"},note)); }
  const chartBase={animation:false,backgroundColor:"transparent",textStyle:{color:"#82929f"},grid:{left:58,right:58,top:28,bottom:38},tooltip:{trigger:"axis",backgroundColor:"#111c25",borderColor:"#28404e",textStyle:{color:"#eef6f7"}},legend:{textStyle:{color:"#82929f"}},xAxis:{type:"category",axisLine:{lineStyle:{color:"#29404d"}},axisLabel:{color:"#71838f"}},yAxis:{type:"value",splitLine:{lineStyle:{color:"#152631"}},axisLabel:{color:"#71838f"}}};

  function Nav({page,setPage}) { return e("aside",{className:"rail"},
    e("div",{className:"brand"},e("div",{className:"brand-mark"},"Q"),e("div",null,e("div",{className:"brand-title"},"QUANT FUND"),e("div",{className:"brand-sub"},"LIVE MONITOR"))),
    e("nav",{className:"nav"},pages.map(([id,label,icon])=>e("button",{key:id,className:page===id?"active":"",onClick:()=>setPage(id)},e("span",{className:"nav-icon"},icon),label))),
    e("div",{className:"rail-foot"},"XGBoost Pseudo-Huber",e("br"),"Inverse Volatility · OOS")
  );}
  function Header({snap}) { const live=snap.quote_connected; return e("header",{className:"topbar"},e("div",null,e("div",{className:"eyebrow"},"Taiwan Equity Systematic Fund"),e("h1",null,"XGBoost 反波動模擬基金"),e("div",{className:"subline"},`訊號 ${snap.signal_date} · 執行 ${snap.execution_date} · 歷史錨點 ${snap.anchor_date}`)),e("div",{className:"status-cluster"},e("span",{className:`pill ${live?"good":"warn"}`},e("i",{className:"dot"}),live?"Shioaji 即時連線":"延遲行情"),e("span",{className:`pill ${snap.valuation_quality==="live"?"good":"warn"}`},`估值 ${snap.valuation_quality}`),e("span",{className:"pill"},new Date(snap.as_of).toLocaleString("zh-TW")))); }
  function Kpis({snap}) { return e("div",{className:"kpis"},
    e(Kpi,{label:"基金總資產",value:money(snap.nav),note:`歷史錨點 ${money(snap.anchor_nav)}`}),
    e(Kpi,{label:"單位淨值",value:num(snap.unit_nav,4),note:"起始單位淨值 10.0000"}),
    e(Kpi,{label:"成立以來報酬",value:pct(snap.return_since_inception),note:"原始資金 NT$30,000,000",tone:cls(snap.return_since_inception)}),
    e(Kpi,{label:"股票曝險",value:pct(snap.equity_exposure),note:`股票 ${money(snap.equity_value)}`}),
    e(Kpi,{label:"現金部位",value:money(snap.cash),note:pct(snap.cash/snap.nav)})
  );}
  function IntradayChart({intraday}) {
    const option=useMemo(()=>{
      const points=intraday?.points||[],day=intraday?.session_date,previousNav=Number(intraday?.previous_close_nav),previousTaiex=Number(intraday?.previous_close_taiex);
      const navs=points.map(x=>x.nav).filter(Number.isFinite),indices=points.map(x=>x.taiex).filter(Number.isFinite);
      const limits=(values,padding)=>values.length?{min:Math.min(...values)*(1-padding),max:Math.max(...values)*(1+padding)}:{};
      const moves=[...navs.map(v=>Number.isFinite(previousNav)&&previousNav>0?Math.abs(v/previousNav-1):0),...indices.map(v=>Number.isFinite(previousTaiex)&&previousTaiex>0?Math.abs(v/previousTaiex-1):0)];
      const required=(Math.max(0,...moves)||0)*1.15;
      const amplitude=([.01,.02,.03,.05,.07,.10].find(level=>level>=required))||.10;
      const centeredLimits=(reference,values,padding)=>Number.isFinite(reference)&&reference>0?{min:reference*(1-amplitude),max:reference*(1+amplitude),splitNumber:4}:limits(values,padding);
      const showPoint=points.length<=1;
      const centerLine=Number.isFinite(previousNav)?{silent:true,symbol:"none",lineStyle:{color:"#d9e4e8",width:2.5,type:"solid",opacity:.9},label:{show:false},data:[{yAxis:previousNav}]}:undefined;
      const gainArea=points.map(x=>[x.time,Number.isFinite(previousNav)?Math.max(x.nav,previousNav):x.nav]);
      const lossArea=points.map(x=>[x.time,Number.isFinite(previousNav)?Math.min(x.nav,previousNav):x.nav]);
      const verticalGradient=stops=>window.echarts?new echarts.graphic.LinearGradient(0,0,0,1,stops):stops[0].color;
      const gainGradient=verticalGradient([{offset:0,color:"rgba(218,68,91,.46)"},{offset:.5,color:"rgba(207,63,84,.32)"},{offset:.85,color:"rgba(196,60,81,.22)"},{offset:1,color:"rgba(188,57,78,.12)"}]);
      const lossGradient=verticalGradient([{offset:0,color:"rgba(29,158,103,.12)"},{offset:.15,color:"rgba(29,158,103,.22)"},{offset:.5,color:"rgba(27,150,97,.32)"},{offset:1,color:"rgba(24,139,90,.45)"}]);
      return {...chartBase,legend:{...chartBase.legend,data:["基金 NAV","TAIEX"]},xAxis:{type:"time",min:day?`${day}T09:00:00+08:00`:null,max:day?`${day}T13:30:00+08:00`:null,axisLine:{show:true,lineStyle:{color:"#527080",width:2}},axisTick:{show:true,lineStyle:{color:"#527080"}},axisLabel:{color:"#8fa2ae",formatter:value=>new Date(value).toLocaleTimeString("zh-TW",{hour:"2-digit",minute:"2-digit",hour12:false})}},yAxis:[{type:"value",name:`基金 NAV  ±${(amplitude*100).toFixed(0)}%`,scale:true,...centeredLimits(previousNav,navs,.005),splitLine:{lineStyle:{color:"#20313a"}},axisLabel:{color:"#8fa2ae",formatter:v=>(v/1e6).toFixed(2)+"M"}},{type:"value",name:"TAIEX",scale:true,...centeredLimits(previousTaiex,indices,.01),splitLine:{show:false},axisLabel:{color:"#a9bd86",formatter:v=>Number(v).toLocaleString("zh-TW",{maximumFractionDigits:0})}}],series:[{name:"基金上漲區",type:"line",showSymbol:false,silent:true,tooltip:{show:false},yAxisIndex:0,lineStyle:{opacity:0},areaStyle:{origin:previousNav,color:gainGradient,opacity:1},data:gainArea},{name:"基金下跌區",type:"line",showSymbol:false,silent:true,tooltip:{show:false},yAxisIndex:0,lineStyle:{opacity:0},areaStyle:{origin:previousNav,color:lossGradient,opacity:1},data:lossArea},{name:"基金 NAV",type:"line",showSymbol:showPoint,symbolSize:8,symbolOffset:showPoint?[0,-6]:[0,0],yAxisIndex:0,lineStyle:{color:"#d7e2e5",width:2.1,shadowColor:"rgba(215,226,229,.18)",shadowBlur:4},itemStyle:{color:"#d7e2e5"},markLine:centerLine,data:points.map(x=>[x.time,x.nav])},{name:"TAIEX",type:"line",showSymbol:showPoint,symbolSize:8,symbolOffset:showPoint?[0,6]:[0,0],yAxisIndex:1,lineStyle:{color:"#adc983",width:1.7,opacity:.72},itemStyle:{color:"#adc983"},data:points.map(x=>[x.time,x.taiex])}]};
    },[intraday]);
    return e(Chart,{option});
  }
  function HistoricalNavChart({perf}) { const option=useMemo(()=>{ const hist=perf.history||[]; return {...chartBase,legend:{...chartBase.legend,data:["基金 NAV"]},xAxis:{...chartBase.xAxis,data:hist.map(x=>x.date)},yAxis:{...chartBase.yAxis,axisLabel:{color:"#71838f",formatter:v=>(v/1e6).toFixed(1)+"M"}},series:[{name:"基金 NAV",type:"line",showSymbol:false,lineStyle:{color:"#35e4c4",width:2},areaStyle:{color:"#35e4c418"},data:hist.map(x=>x.nav)}]}; },[perf]); return e(Chart,{option}); }
  function Contributors({holdings}) { const rows=[...holdings].sort((a,b)=>Math.abs(b.pnl_since_anchor)-Math.abs(a.pnl_since_anchor)).slice(0,6); return e("div",{className:"metric-list"},rows.map(r=>e("div",{className:"metric-row",key:r.stock_id},e("span",null,e("span",{className:"stock-code"},r.stock_id)," ",r.stock_name),e("strong",{className:cls(r.pnl_since_anchor)},money(r.pnl_since_anchor))))); }
  function Overview({snap,intraday}) { const clearing=intraday?.phase==="clearing",m=clearing?null:snap.market,status=clearing?"待開盤":intraday?.data_status==="closing"?"收盤資料確認中":intraday?.data_status==="delayed"?"行情回補中":"",move=m?.reference?m.price-m.reference:null,moveTone=cls(move),moveText=move==null?"—":move>0?`▲ ${num(Math.abs(move),2)}`:move<0?`▼ ${num(Math.abs(move),2)}`:`— ${num(0,2)}`; return e(React.Fragment,null,e(Kpis,{snap}),e("div",{className:"grid2"},e(Card,{title:"當日基金 NAV 與 TAIEX",note:`${intraday?.session_date||"—"} · 台灣時間 09:00–13:30${status?` · ${status}`:""}`},e(IntradayChart,{intraday})),e("div",{className:"card market-box"},e("div",{className:"section-title"},"市場即時狀態"),m?e(React.Fragment,null,e("div",{className:"market-value-row"},e("div",{className:`market-value ${moveTone}`},num(m.price,2)),e("div",{className:`market-points ${moveTone}`},moveText)),e("div",{className:"market-grid"},e("div",{className:"mini"},e("span",null,"指數漲跌幅"),e("strong",{className:moveTone},m.reference?pct(m.price/m.reference-1):"—")),e("div",{className:"mini"},e("span",null,"資料來源"),e("strong",null,m.source)),e("div",{className:"mini"},e("span",null,"累計成交額"),e("strong",null,yi(m.amount))),e("div",{className:"mini"},e("span",null,"行情時間"),e("strong",null,new Date(m.timestamp).toLocaleString("zh-TW"))))):e("div",{className:"empty"},clearing?"待開盤：即時行情將於 09:00 開始":"TAIEX 行情等待中"))),e("div",{className:"grid2"},e(Card,{title:"損益貢獻",note:"相對歷史錨點"},e(Contributors,{holdings:snap.holdings})),e(Card,{title:"監控事項"},e("div",{className:"alert"},"●",e("span",null,snap.quote_message,"。行情覆蓋率 ",pct(snap.quote_coverage),"；圖表只在新報價抵達時增加資料點。"))))); }
  function HoldingsTable({rows}) { return e("div",{className:"table-wrap"},e("table",null,e("thead",null,e("tr",null,["排名","股票","最新價","漲跌","股數","市值","即時權重","目標權重","偏離","買／賣"].map(x=>e("th",{key:x},x)))),e("tbody",null,rows.map(r=>e("tr",{key:r.stock_id},e("td",null,e("span",{className:"rank"},r.rank)),e("td",null,e("span",{className:"stock-code"},r.stock_id)," ",e("span",{className:"stock-name"},r.stock_name)),e("td",null,num(r.last_price,2)),e("td",{className:cls(r.day_change)},pct(r.day_change)),e("td",null,num(r.shares,0)),e("td",null,money(r.market_value)),e("td",null,pct(r.live_weight),e("span",{className:"bar"},e("i",{style:{width:`${Math.min((r.live_weight||0)*1000,100)}%`}}))),e("td",null,pct(r.target_weight)),e("td",{className:cls(r.live_weight-r.target_weight)},pct(r.live_weight-r.target_weight)),e("td",null,r.bid&&r.ask?`${num(r.bid)} / ${num(r.ask)}`:"—")))))); }
  function Holdings({snap}) { const [sort,setSort]=useState("rank"); const rows=useMemo(()=>[...snap.holdings].sort((a,b)=>sort==="market"?b.market_value-a.market_value:sort==="pnl"?b.pnl_since_anchor-a.pnl_since_anchor:a.rank-b.rank),[snap,sort]); return e(React.Fragment,null,e(Kpis,{snap}),e(Card,{title:"當前持股明細",note:`${rows.length} 檔 · 報價覆蓋 ${pct(snap.quote_coverage)}`},e("div",{className:"toolbar"},e("span",{className:"section-note"},"排序"),e("select",{className:"select",value:sort,onChange:x=>setSort(x.target.value)},e("option",{value:"rank"},"模型排名"),e("option",{value:"market"},"持股市值"),e("option",{value:"pnl"},"損益貢獻"))),e(HoldingsTable,{rows}))); }
  function Performance({perf,onRefresh,refreshing,updatedAt}) { const monthly=perf.monthly_returns||[]; const barOption=useMemo(()=>({...chartBase,xAxis:{...chartBase.xAxis,data:monthly.map(x=>x.month)},yAxis:{...chartBase.yAxis,axisLabel:{color:"#71838f",formatter:v=>(v*100).toFixed(0)+"%"}},series:[{type:"bar",data:monthly.map(x=>({value:x.return,itemStyle:{color:x.return>=0?"#ff6478":"#4ce395"}}))}]}),[perf]); const m=perf.metrics||{}; return e(React.Fragment,null,e("div",{className:"toolbar performance-refresh"},e("span",{className:"section-note"},`資料更新：${updatedAt?updatedAt.toLocaleString("zh-TW"):"—"}`),e("button",{className:"action-button",disabled:refreshing,onClick:onRefresh},refreshing?"更新中…":"更新資料")),e("div",{className:"grid3"},e(Kpi,{label:"歷史 CAGR",value:pct(m.cagr),note:"OOS 年化"}),e(Kpi,{label:"歷史 Sharpe",value:num(m.sharpe),note:"風險調整報酬"}),e(Kpi,{label:"最大回撤",value:pct(m.max_drawdown),note:"OOS 區間",tone:"negative"})),e(Card,{title:"基金歷史 NAV",note:"手動更新；不含 TAIEX"},e(HistoricalNavChart,{perf})),e(Card,{title:"每月報酬",note:"歷史 materialized artifact"},e(Chart,{option:barOption,small:true}))); }
  function History({history,orders}) {
    const dates=[...new Set(history.map(x=>x.signal_date))];
    const [date,setDate]=useState(dates[0]||"");
    const rows=history.filter(x=>x.signal_date===date);
    const recent=(orders||[]).slice(0,50);
    const holdingRows=rows.map(r=>e("tr",{key:r.stock_id},
      e("td",null,e("span",{className:"rank"},r.rank)),
      e("td",null,e("span",{className:"stock-code"},r.stock_id)," ",r.stock_name),
      e("td",null,r.signal),e("td",null,pct(r.target_weight)),e("td",null,r.signal_date)
    ));
    const orderRows=recent.map((r,i)=>e("tr",{key:`${r.execution_date}-${r.stock_id}-${i}`},
      e("td",null,r.execution_date),
      e("td",null,e("span",{className:"stock-code"},r.stock_id)," ",r.stock_name),
      e("td",{className:r.side==="BUY"?"positive":"negative"},r.side),
      e("td",null,num(r.quantity,0)),e("td",null,num(r.executed_price,2)),
      e("td",null,money(r.notional)),e("td",null,money(r.total_cost))
    ));
    const table=(head,body)=>e("div",{className:"table-wrap"},e("table",null,
      e("thead",null,e("tr",null,head.map(x=>e("th",{key:x},x)))),e("tbody",null,body)
    ));
    return e(React.Fragment,null,
      e(Card,{title:"歷史持股與換倉",note:"選擇訊號月份查看目標組合"},
        e("div",{className:"toolbar"},
          e("select",{className:"select",value:date,onChange:x=>setDate(x.target.value)},dates.map(d=>e("option",{key:d,value:d},d))),
          e("span",{className:"pill"},`${rows.length} 檔`)
        ),
        table(["排名","股票","動作","目標權重","訊號日"],holdingRows)
      ),
      e(Card,{title:"模擬成交紀錄",note:"最近 50 筆；沿用 OOS 成本與滑價"},
        table(["執行日","股票","方向","股數","成交價","成交金額","總成本"],orderRows)
      )
    );
  }
  function Risk({risk,snap}) {
    if(!risk)return e("div",{className:"empty"},"風險資料載入中");
    const cards=[["股票曝險",pct(risk.equity_exposure)],["現金比例",pct(risk.cash_ratio)],["最大單股",pct(risk.largest_weight)],["前五大持股",pct(risk.top5_weight)],["歷史波動率",pct(risk.historical_realized_volatility)],["歷史換手率",num(risk.turnover,2)+"x"]];
    const regime=risk.taiex_above_ma60;
    const quality=e("div",{className:"metric-list"},
      e("div",{className:"metric-row"},e("span",null,"Shioaji 連線"),e("strong",{className:snap.quote_connected?"status-good":"warning"},snap.quote_connected?"正常":"延遲模式")),
      e("div",{className:"metric-row"},e("span",null,"持股行情覆蓋"),e("strong",null,pct(risk.quote_coverage))),
      e("div",{className:"metric-row"},e("span",null,"估值品質"),e("strong",null,risk.valuation_quality)),
      e("div",{className:"metric-row"},e("span",null,"歷史最大回撤"),e("strong",{className:"negative"},pct(risk.historical_max_drawdown)))
    );
    const state=e(React.Fragment,null,
      e("div",{className:"metric-list"},
        e("div",{className:"metric-row"},e("span",null,"TAIEX / MA60"),e("strong",{className:regime==null?"warning":regime?"positive":"negative"},regime==null?"資料不足":regime?"多頭區間":"風控區間")),
        e("div",{className:"metric-row"},e("span",null,"TAIEX MA60"),e("strong",null,num(risk.taiex_ma60,2)))
      ),
      e("div",{className:"alert"},"●",e("span",null,`最新正式訊號 ${snap.signal_date}。下一個月度訊號物化後，系統每分鐘偵測並沿用 Top15／Hold30、反波動權重與既有成本規則接續。`))
    );
    return e(React.Fragment,null,
      e("div",{className:"kpis"},cards.map(([a,b])=>e(Kpi,{key:a,label:a,value:b,note:"風險監控"}))),
      e("div",{className:"grid2"},e(Card,{title:"資料與估值品質"},quality),e(Card,{title:"策略狀態"},state))
    );
  }
  function App(){ const [page,setPage]=useState("overview"),[snap,setSnap]=useState(null),[perf,setPerf]=useState(null),[intraday,setIntraday]=useState(null),[history,setHistory]=useState([]),[orders,setOrders]=useState([]),[risk,setRisk]=useState(null),[perfUpdated,setPerfUpdated]=useState(null),[perfRefreshing,setPerfRefreshing]=useState(false),[error,setError]=useState(null); const refreshPerformance=()=>{setPerfRefreshing(true);api("/api/fund/performance").then(p=>{setPerf(p);setPerfUpdated(new Date())}).catch(x=>setError(x.message)).finally(()=>setPerfRefreshing(false))}; useEffect(()=>{let stopped=false,ws=null,retryTimer=null,retry=1000,syncToken=0;const syncIntraday=()=>{const token=++syncToken;return api("/api/fund/intraday").then(payload=>{if(!stopped&&token===syncToken)setIntraday(payload)})};const appendPoint=point=>setIntraday(previous=>{if(!previous||previous.session_date!==point.session_date)return previous;const points=previous.points||[],index=points.findIndex(item=>item.time===point.time);if(index>=0){if(points[index].quote_revision===point.quote_revision&&points[index].nav===point.nav&&points[index].taiex===point.taiex)return previous;const next=[...points];next[index]=point;return {...previous,points:next}}return {...previous,points:[...points,point].sort((a,b)=>a.time.localeCompare(b.time))}});const connect=()=>{if(stopped)return;syncIntraday().finally(()=>{if(stopped)return;const protocol=location.protocol==="https:"?"wss":"ws";ws=new WebSocket(`${protocol}://${location.host}/ws/live`);ws.onopen=()=>{retry=1000};ws.onmessage=x=>{const s=JSON.parse(x.data);setSnap(s);setIntraday(previous=>{if(s.intraday_clear)return {session_date:s.session_date,market_open:"09:00",market_close:"13:30",phase:"clearing",data_status:"waiting",points:[]};if(!previous||previous.session_date!==s.session_date||previous.phase!==s.market_phase){syncIntraday();return previous}return previous});if(s.intraday_point)appendPoint(s.intraday_point)};ws.onclose=()=>{if(stopped)return;retryTimer=setTimeout(connect,retry);retry=Math.min(retry*2,30000)}})};Promise.all([api("/api/fund/overview"),api("/api/fund/performance"),api("/api/fund/intraday"),api("/api/fund/holding-history"),api("/api/fund/order-history"),api("/api/fund/risk")]).then(([s,p,i,h,o,r])=>{if(stopped)return;setSnap(s);setPerf(p);setIntraday(i);setHistory(h);setOrders(o);setRisk(r);setPerfUpdated(new Date());connect()}).catch(x=>setError(x.message));const refresh=setInterval(()=>api("/api/fund/risk").then(setRisk),30000);return()=>{stopped=true;if(ws)ws.close();if(retryTimer)clearTimeout(retryTimer);clearInterval(refresh)}},[]); if(error)return e("div",{className:"loading negative"},error); if(!snap||!perf)return e("div",{className:"loading"},"INITIALIZING FUND MONITOR"); let content=page==="overview"?e(Overview,{snap,intraday}):page==="holdings"?e(Holdings,{snap}):page==="performance"?e(Performance,{perf,onRefresh:refreshPerformance,refreshing:perfRefreshing,updatedAt:perfUpdated}):page==="history"?e(History,{history,orders}):e(Risk,{risk,snap}); return e("div",{className:"app"},e(Nav,{page,setPage}),e("main",{className:"main"},e(Header,{snap}),content)); }
  ReactDOM.createRoot(document.getElementById("root")).render(e(App));
})();
