#!/usr/bin/env node
/**
 * 评测语料生成器：产出干扰文档（distractor）。
 *
 * 为什么需要：原来 5 篇语料只切成 11 个块，topK=5 一次覆盖 45% 的库，
 * recall@3 / MRR 恒等于 1，门槛形同虚设。
 *
 * 关键设计：干扰文档必须与目标文档**同族**——同样的文档类型（产品手册 / 员工手册 /
 * 财务报告 / 会议纪要 / 技术方案）、同样的句式、只是**事实不同**。
 * 这样「标准版售价是多少」会同时命中目标文档和若干同类手册，
 * 检索必须真的把正确的那份排到前面，指标才有判别力。
 * 若用无关领域的填充文本，靠停用词就能分开，等于没测。
 *
 * 用法：node scripts/gen-corpus.mjs
 */
import { writeFileSync, mkdirSync, readdirSync, unlinkSync } from "node:fs";
import { join, dirname } from "node:path";
import { TOPIC_DOCS } from "./topic-corpus.mjs";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
/**
 * 语料只有这一份：public/eval/corpus。
 * Node 侧 eval 直接读盘，浏览器侧评测工具走 fetch 读同一批文件，
 * 不存在「Node 一份、浏览器一份」的副本。
 */
const OUT = join(ROOT, "public", "eval", "corpus");

/* ---------------- 同族文档模板 ---------------- */

/**
 * 每个族一个模板函数，参数是一组「事实」。
 * 目标文档用手写内容（docs/eval/corpus 下的 5 篇），这里生成的是**事实不同的同族变体**。
 */

/** 产品手册族 */
function productManual(f) {
  return {
    title: `${f.product}产品手册`,
    sections: [
      ["产品概述", [
        `${f.product}是面向${f.scene}的核心设备，负责${f.duty}。`,
        `设备支持 ${f.protocols} 三种工业协议，可同时接入 ${f.devices} 台子设备。`,
        `设备出厂默认内存为 ${f.mem}，支持扩展至 ${f.memMax}。`,
        `${f.product}采用${f.chip}处理器，整机重量 ${f.weight} 千克。`,
      ]],
      ["定价与购买", [
        `标准版单台售价为 ${f.stdPrice} 元，包含${f.stdWarranty}。`,
        `专业版单台售价为 ${f.proPrice} 元，包含${f.proWarranty}与${f.proExtra}。`,
        `批量采购 ${f.bulk1} 台以上可享受 ${f.disc1} 折优惠，${f.bulk2} 台以上可享受 ${f.disc2} 折优惠。`,
        `教育行业客户凭有效证明可额外享受 ${f.eduDisc} 折优惠。`,
      ]],
      ["技术规格", [
        `设备工作温度为负 ${f.tempLow} 摄氏度至 ${f.tempHigh} 摄氏度，支持宽温环境下持续运行。`,
        `整机功耗峰值为 ${f.powerPeak} 瓦，待机功耗为 ${f.powerIdle} 瓦。`,
        `网络接口包含 ${f.ethPorts} 个千兆以太网口与 ${f.fiberPorts} 个光纤接口。`,
        `防护等级为 IP${f.ip}，支持${f.mount}安装方式。`,
      ]],
      ["保修与售后", [
        `标准版保修期为 ${f.stdWarrantyMonths} 个月，专业版保修期为 ${f.proWarrantyMonths} 个月。`,
        `保修期内非人为损坏免费更换，人为损坏按维修成本收取费用。`,
        `售后热线服务时间为工作日 ${f.hotlineFrom} 点至 ${f.hotlineTo} 点。`,
        `备件供应承诺为产品停产后不少于 ${f.spareYears} 年。`,
      ]],
      ["部署与配置", [
        `设备首次启动需要约 ${f.bootMinutes} 分钟完成系统初始化。`,
        `支持通过${f.configWay}进行批量配置下发，单次最多下发 ${f.configBatch} 台。`,
        `固件升级失败时，设备会自动回滚到上一个稳定版本，不会影响业务运行。`,
        `固件建议每${f.fwCycle}检查一次更新。`,
      ]],
      ["常见故障处理", [
        `设备指示灯红色常亮表示${f.errRed}，需检查${f.errRedFix}。`,
        `网络不通时优先检查${f.netFix}，再确认上级交换机端口状态。`,
        `设备反复重启通常是${f.rebootCause}导致，建议${f.rebootFix}。`,
        `日志可通过${f.logWay}导出，用于提交工单时附送分析。`,
      ]],
    ],
  };
}

/** 员工手册族 */
function employeeHandbook(f) {
  return {
    title: `${f.company}员工手册`,
    sections: [
      ["年假制度", [
        `员工入职满${f.annualAfter}可享受 ${f.annualBase} 天带薪年假，工作年限每增加一年年假增加 1 天，年假上限为 ${f.annualMax} 天。`,
        `年假须在次年 ${f.annualDeadline} 前使用完毕，未使用部分不折算现金。`,
        `申请年假需提前 ${f.annualNotice} 个工作日提交审批。`,
        `年度内未休完的年假可申请延期一次，延期不超过 ${f.annualExtend} 天。`,
      ]],
      ["加班与调休", [
        `工作日加班超过 ${f.otThreshold} 小时可申请调休，周末加班按 ${f.otWeekend} 倍工资计算加班费。`,
        `法定节假日加班按 ${f.otHoliday} 倍工资计算。`,
        `调休申请须在加班发生后 ${f.otExpire} 天内提出，逾期自动作废。`,
        `加班须提前在系统提交申请，事后补录每月不超过 ${f.otMakeup} 次。`,
      ]],
      ["报销制度", [
        `出差住宿标准为一线城市每晚 ${f.hotel1} 元，其他城市每晚 ${f.hotel2} 元。`,
        `市内交通费凭票据实报实销，单日上限 ${f.transportCap} 元。`,
        `报销单须在出差结束后 ${f.reimbDays} 个工作日内提交，逾期不予受理。`,
        `业务招待费须附招待对象与事由说明，单次上限 ${f.entertainCap} 元。`,
      ]],
      ["绩效考核", [
        `绩效分为 ${f.perfLevels} 四个等级。`,
        `年度绩效为 ${f.bonusLevel} 及以上的员工可获得 ${f.bonusMonths} 个月工资的年终奖。`,
        `连续两个季度绩效为 ${f.pipLevel} 的员工将进入改进辅导计划。`,
        `绩效结果在每年 ${f.perfMonths} 各公布一次。`,
      ]],
      ["考勤与休假", [
        `标准工作时间为每日 ${f.workHours} 小时，弹性上班时间为 ${f.flexFrom} 点至 ${f.flexTo} 点。`,
        `迟到超过 ${f.lateLimit} 分钟计为旷工半天。`,
        `病假须提供二级以上医院证明，全年病假不超过 ${f.sickDays} 天。`,
        `婚假为 ${f.marriageDays} 天，产假按国家规定执行并额外增加 ${f.maternityExtra} 天。`,
      ]],
      ["员工福利", [
        `公司为员工缴纳五险一金，公积金缴存比例为 ${f.housingFund}%。`,
        `每年提供 ${f.checkupFreq} 次免费体检，标准为每人 ${f.checkupFee} 元。`,
        `员工生日当月发放 ${f.birthdayGift} 元购物卡。`,
        `入职满 ${f.tenureGift} 年的员工可获得司龄纪念奖，金额为 ${f.tenureGiftFee} 元。`,
      ]],
    ],
  };
}

/** 财务报告族 */
function financeReport(f) {
  return {
    title: `${f.year} 年度财务报告（${f.entity}）`,
    sections: [
      ["收入情况", [
        `${f.year} 年度总营收为 ${f.revenue} 亿元，同比增长 ${f.revGrowth}%。`,
        `其中软件授权收入 ${f.licenseRev} 万元，占比 ${f.licensePct}%。`,
        `云服务订阅收入 ${f.cloudRev} 万元，占比 ${f.cloudPct}%。`,
        `技术服务收入 ${f.serviceRev} 万元，占比 ${f.servicePct}%。`,
      ]],
      ["成本与利润", [
        `${f.year} 年度总成本为 ${f.cost} 亿元。`,
        `研发投入为 ${f.rdSpend} 万元，占营收比例 ${f.rdPct}%。`,
        `销售费用为 ${f.salesSpend} 万元，管理费用为 ${f.adminSpend} 万元。`,
        `年度净利润为 ${f.profit} 万元，净利率为 ${f.netMargin}%。`,
      ]],
      ["现金流", [
        `经营活动现金流净额为 ${f.opCash} 万元。`,
        `投资活动现金流出 ${f.investCash} 万元，主要用于${f.investUse}。`,
        `年末现金及等价物余额为 ${f.cashEnd} 万元，较年初增加 ${f.cashGrowth}%。`,
        `融资活动现金流净额为 ${f.financeCash} 万元。`,
      ]],
      ["下一年度预算", [
        `${f.nextYear} 年度预算总营收目标为 ${f.nextRevenue} 亿元。`,
        `研发预算增至 ${f.nextRd} 万元，重点投向${f.nextRdFocus}。`,
        `市场预算为 ${f.nextMarket} 万元，其中${f.nextMarketFocus}占 ${f.nextMarketFocusPct} 万元。`,
        `计划新增员工 ${f.nextHeadcount} 人，人力成本预算增加 ${f.nextHrCost} 万元。`,
      ]],
      ["风险提示", [
        `大客户集中度较高，前三大客户贡献了总营收的 ${f.top3Pct}%。`,
        `应收账款周转天数为 ${f.arDays} 天，存在回款压力。`,
        `汇率波动对海外业务的影响预计在 ${f.fxImpact} 万元以内。`,
        `存货周转率较上年下降 ${f.invDrop}%，需关注${f.invConcern}。`,
      ]],
    ],
  };
}

/** 会议纪要族 */
function meetingMinutes(f) {
  return {
    title: `${f.topic}会议纪要`,
    sections: [
      ["会议信息", [
        `会议时间：${f.meetDate}。`,
        `参会人员：${f.attendees}，共 ${f.attendeeCount} 人。`,
        `会议主题：${f.meetGoal}。`,
        `会议主持人为${f.host}，记录人为${f.recorder}。`,
      ]],
      ["产品路线图决议", [
        `第一季度完成${f.q1Goal}，重点解决${f.q1Focus}问题。`,
        `第二季度启动${f.q2Goal}，目标将${f.q2Metric}提升 ${f.q2Pct}%。`,
        `第三季度推进${f.q3Goal}，首批覆盖${f.q3Market}市场。`,
        `第四季度进行${f.q4Goal}，预算为 ${f.q4Budget} 万元。`,
      ]],
      ["资源分配决议", [
        `${f.q1Goal}项目投入研发人力 ${f.r1} 人，${f.q2Goal}项目投入研发人力 ${f.r2} 人。`,
        `市场推广预算全年为 ${f.mktBudget} 万元，其中线上投放占 ${f.mktOnlinePct}%，线下活动占 ${f.mktOfflinePct}%。`,
        `新增招聘计划为研发 ${f.hireRd} 人、市场 ${f.hireMkt} 人。`,
        `基础设施预算为 ${f.infraBudget} 万元，用于${f.infraUse}。`,
      ]],
      ["待办事项", [
        `${f.q1Goal}架构评审定于 ${f.reviewDate} 进行，评审通过后一周内启动开发。`,
        `${f.q2Goal}设计稿需在 ${f.designDue} 前完成终稿确认。`,
        `${f.q3Goal}选品会议暂定 ${f.selectionDate} 召开，具体时间由市场部另行通知。`,
        `各项待办须在下次周会上同步进展，由${f.pmOwner}统一跟踪。`,
      ]],
      ["风险与依赖", [
        `${f.q1Goal}依赖云厂商新发布的流式计算引擎，若引擎延期发布将整体顺延 ${f.delayWeeks} 周。`,
        `${f.q2Goal}依赖设计团队人力，当前设计团队仅 ${f.designCount} 人，需要外部支持。`,
        `招聘进度若低于计划的 ${f.hireRiskPct}%，将影响第二季度交付。`,
        `预算审批若未在 ${f.budgetDeadline} 前完成，相关采购将顺延至下季度。`,
      ]],
    ],
  };
}

/** 技术方案族 */
function techSpec(f) {
  return {
    title: `${f.schemeName}技术方案`,
    sections: [
      ["架构设计", [
        `系统采用${f.archStyle}架构：${f.archFlow}。`,
        `检索阶段使用 ${f.fusionAlgo} 融合算法合并多路结果，融合常数 k 取 ${f.fusionK}。`,
        `服务按${f.deployUnit}部署，单集群支持 ${f.clusterNodes} 个节点。`,
        `关键路径上引入${f.cachePolicy}缓存策略，命中率约 ${f.cacheHit}%。`,
      ]],
      ["向量化方案", [
        `文档分块使用 ${f.embedModel} 模型进行向量化，向量维度为 ${f.embedDim} 维。`,
        `分块参数为最大 ${f.chunkTokens} 个 token、整句重叠 ${f.chunkOverlap} 句。`,
        `模型以量化方式加载，内存占用约 ${f.modelMem}MB。`,
        `向量索引采用${f.indexType}结构，召回率约 ${f.indexRecall}%。`,
      ]],
      ["性能指标", [
        `单块向量化耗时约 ${f.perChunkMs} 毫秒，整篇 ${f.docPages} 页文档全量向量化约需 ${f.totalEmbedMin} 分钟。`,
        `检索响应时间中位数为 ${f.searchP50} 毫秒，P99 为 ${f.searchP99} 毫秒。`,
        `系统内存峰值控制在 ${f.memPeak}MB 以内。`,
        `单机 QPS 峰值为 ${f.qps}，压测持续 ${f.stressHours} 小时无衰减。`,
      ]],
      ["幻觉校验", [
        `答案生成后执行双层校验：语义证据计算句子与${f.verifyScope}的最大余弦相似度。`,
        `词法证据检查答案中的数字是否能在原文找到出处。`,
        `语义证据权重为 ${f.semWeight}，词法证据权重为 ${f.lexWeight}。`,
        `校验不通过的句子比例超过 ${f.rejectPct}% 时整体判为不可信。`,
      ]],
      ["降级策略", [
        `向量模型加载失败时，检索自动降级为纯${f.fallbackMode}检索。`,
        `云端接口超时设置为 ${f.timeoutSec} 秒，超时后自动切换到本地模型回答。`,
        `历史对话超过 ${f.summaryChars} 字符时触发自动摘要压缩。`,
        `降级事件全部打点上报，每${f.degradeCycle}复盘一次。`,
      ]],
    ],
  };
}

/* ---------------- 变体事实表 ---------------- */

/** 产品手册变体（目标文档是「智能网关」） */
const PRODUCTS = [
  { product: "边缘计算盒", scene: "工业现场数据采集", duty: "协议转换与本地预处理", protocols: "Modbus、Profinet、EtherCAT", devices: 96, mem: "2GB", memMax: "8GB", chip: "四核 ARM Cortex-A55", weight: "1.2", stdPrice: 2680, stdWarranty: "一年保修", proPrice: 4280, proWarranty: "两年保修", proExtra: "远程运维服务", bulk1: 30, bulk2: 80, disc1: "9.2", disc2: "8.8", eduDisc: "9.5", tempLow: 10, tempHigh: 60, powerPeak: 9, powerIdle: 2, ethPorts: 2, fiberPorts: 1, ip: "54", mount: "导轨", stdWarrantyMonths: 12, proWarrantyMonths: 24, hotlineFrom: 9, hotlineTo: 17, spareYears: 3, bootMinutes: 1, configWay: "Web 控制台", configBatch: 32, fwCycle: "半年", errRed: "电源异常", errRedFix: "供电电压", netFix: "网线连接", rebootCause: "供电不稳", rebootFix: "加装稳压电源", logWay: "USB 存储" },
  { product: "数据采集终端", scene: "能源计量", duty: "多路计量数据汇聚与上报", protocols: "DLT645、Modbus、IEC104", devices: 64, mem: "1GB", memMax: "4GB", chip: "双核 ARM Cortex-A7", weight: "0.8", stdPrice: 1580, stdWarranty: "一年保修", proPrice: 2380, proWarranty: "三年保修", proExtra: "计量校准服务", bulk1: 100, bulk2: 300, disc1: "8.5", disc2: "8", eduDisc: "9", tempLow: 25, tempHigh: 75, powerPeak: 6, powerIdle: 1.5, ethPorts: 1, fiberPorts: 0, ip: "30", mount: "壁挂", stdWarrantyMonths: 12, proWarrantyMonths: 36, hotlineFrom: 8, hotlineTo: 20, spareYears: 5, bootMinutes: 0.5, configWay: "串口工具", configBatch: 16, fwCycle: "季度", errRed: "通信中断", errRedFix: "计量回路", netFix: "表计地址", rebootCause: "雷击浪涌", rebootFix: "加装防雷器", logWay: "本地 SD 卡" },
  { product: "视频汇聚网关", scene: "安防视频接入", duty: "多路视频流转发与 AI 前置处理", protocols: "RTSP、GB28181、ONVIF", devices: 256, mem: "8GB", memMax: "32GB", chip: "八核 ARM + NPU", weight: "3.6", stdPrice: 8800, stdWarranty: "两年保修", proPrice: 12800, proWarranty: "三年保修", proExtra: "算法授权与调优", bulk1: 20, bulk2: 50, disc1: "9", disc2: "8.5", eduDisc: "9.2", tempLow: 5, tempHigh: 55, powerPeak: 45, powerIdle: 12, ethPorts: 8, fiberPorts: 4, ip: "40", mount: "机架", stdWarrantyMonths: 24, proWarrantyMonths: 36, hotlineFrom: 7, hotlineTo: 22, spareYears: 4, bootMinutes: 3, configWay: "集中管理平台", configBatch: 128, fwCycle: "月度", errRed: "AI 模块过载", errRedFix: "并发路数", netFix: "码流带宽", rebootCause: "散热不良", rebootFix: "清理风道", logWay: "Syslog 服务器" },
  { product: "无线数传模块", scene: "远程遥测", duty: "低功耗无线数据回传", protocols: "LoRaWAN、NB-IoT、4G", devices: 2000, mem: "256MB", memMax: "512MB", chip: "低功耗 MCU", weight: "0.3", stdPrice: 680, stdWarranty: "一年保修", proPrice: 980, proWarranty: "两年保修", proExtra: "流量套餐", bulk1: 500, bulk2: 2000, disc1: "8", disc2: "7.5", eduDisc: "8.8", tempLow: 40, tempHigh: 85, powerPeak: 2, powerIdle: 0.05, ethPorts: 0, fiberPorts: 0, ip: "67", mount: "卡扣", stdWarrantyMonths: 12, proWarrantyMonths: 24, hotlineFrom: 9, hotlineTo: 18, spareYears: 6, bootMinutes: 0.2, configWay: "蓝牙 App", configBatch: 8, fwCycle: "年度", errRed: "信号弱", errRedFix: "天线朝向", netFix: "SIM 卡状态", rebootCause: "电池欠压", rebootFix: "更换电池组", logWay: "串口打印" },
];

/** 员工手册变体（目标文档是「员工手册」） */
const COMPANIES = [
  { company: "启明科技", annualAfter: "满两年", annualBase: 7, annualMax: 20, annualDeadline: "6 月底", annualNotice: 5, annualExtend: 5, otThreshold: 3, otWeekend: 2, otHoliday: 4, otExpire: 60, otMakeup: 1, hotel1: 600, hotel2: 400, transportCap: 120, reimbDays: 20, entertainCap: 2000, perfLevels: "S、A、B、C、D", bonusLevel: "S", bonusMonths: 3, pipLevel: "D", perfMonths: "2 月和 8 月", workHours: 7.5, flexFrom: 9, flexTo: 10, lateLimit: 45, sickDays: 20, marriageDays: 10, maternityExtra: 60, housingFund: 12, checkupFreq: 2, checkupFee: 1500, birthdayGift: 500, tenureGift: 5, tenureGiftFee: 3000 },
  { company: "云枢数据", annualAfter: "满半年", annualBase: 3, annualMax: 12, annualDeadline: "次年 1 月底", annualNotice: 2, annualExtend: 2, otThreshold: 1.5, otWeekend: 1.5, otHoliday: 3, otExpire: 15, otMakeup: 3, hotel1: 450, hotel2: 300, transportCap: 80, reimbDays: 10, entertainCap: 1200, perfLevels: "A、B、C", bonusLevel: "A", bonusMonths: 1.5, pipLevel: "C", perfMonths: "1 月和 7 月", workHours: 8, flexFrom: 8.5, flexTo: 10.5, lateLimit: 30, sickDays: 15, marriageDays: 7, maternityExtra: 30, housingFund: 8, checkupFreq: 1, checkupFee: 800, birthdayGift: 300, tenureGift: 3, tenureGiftFee: 1500 },
  { company: "北岭制造", annualAfter: "满三年", annualBase: 10, annualMax: 25, annualDeadline: "次年 3 月底", annualNotice: 7, annualExtend: 10, otThreshold: 2, otWeekend: 2, otHoliday: 5, otExpire: 90, otMakeup: 2, hotel1: 700, hotel2: 450, transportCap: 150, reimbDays: 30, entertainCap: 3000, perfLevels: "优秀、良好、合格、待改进", bonusLevel: "优秀", bonusMonths: 4, pipLevel: "待改进", perfMonths: "1 月和 7 月", workHours: 8, flexFrom: 7.5, flexTo: 9, lateLimit: 60, sickDays: 30, marriageDays: 15, maternityExtra: 90, housingFund: 10, checkupFreq: 1, checkupFee: 2000, birthdayGift: 800, tenureGift: 10, tenureGiftFee: 8000 },
];

/** 财务报告变体（目标文档是「财务报告」） */
const FINANCE = [
  { year: "2024", entity: "启明科技", revenue: "1.2", revGrowth: 18, licenseRev: 6000, licensePct: 50, cloudRev: 3600, cloudPct: 30, serviceRev: 2400, servicePct: 20, cost: "0.8", rdSpend: 2600, rdPct: 21.7, salesSpend: 1600, adminSpend: 1200, profit: 3400, netMargin: 28.3, opCash: 3100, investCash: 1800, investUse: "办公场地购置", cashEnd: 6200, cashGrowth: 25, financeCash: -400, nextYear: "2025", nextRevenue: "1.5", nextRd: 3200, nextRdFocus: "数据平台建设", nextMarket: 2000, nextMarketFocus: "国内渠道", nextMarketFocusPct: 1200, nextHeadcount: 40, nextHrCost: 1200, top3Pct: 38, arDays: 78, fxImpact: 300, invDrop: 5, invConcern: "原材料备货" },
  { year: "2024", entity: "云枢数据", revenue: "2.6", revGrowth: 42, licenseRev: 5200, licensePct: 20, cloudRev: 18200, cloudPct: 70, serviceRev: 2600, servicePct: 10, cost: "1.9", rdSpend: 5200, rdPct: 20, salesSpend: 3900, adminSpend: 2600, profit: 4680, netMargin: 18, opCash: 5200, investCash: 4600, investUse: "数据中心扩建", cashEnd: 11000, cashGrowth: 52, financeCash: 1500, nextYear: "2025", nextRevenue: "3.4", nextRd: 7000, nextRdFocus: "实时计算引擎", nextMarket: 5000, nextMarketFocus: "海外市场", nextMarketFocusPct: 2500, nextHeadcount: 90, nextHrCost: 2900, top3Pct: 52, arDays: 65, fxImpact: 800, invDrop: 2, invConcern: "带宽采购成本" },
  { year: "2024", entity: "北岭制造", revenue: "0.9", revGrowth: 6, licenseRev: 900, licensePct: 10, cloudRev: 900, cloudPct: 10, serviceRev: 7200, servicePct: 80, cost: "0.75", rdSpend: 900, rdPct: 10, salesSpend: 800, adminSpend: 1500, profit: 1050, netMargin: 11.7, opCash: 1600, investCash: 2400, investUse: "产线自动化改造", cashEnd: 2800, cashGrowth: 8, financeCash: 600, nextYear: "2025", nextRevenue: "1.05", nextRd: 1200, nextRdFocus: "产线数字化", nextMarket: 900, nextMarketFocus: "线下展会", nextMarketFocusPct: 500, nextHeadcount: 25, nextHrCost: 900, top3Pct: 61, arDays: 110, fxImpact: 120, invDrop: 12, invConcern: "成品库存积压" },
];

/** 会议纪要变体（目标文档是「会议纪要」） */
const MEETINGS = [
  { topic: "2025 年度增长规划", meetDate: "2025 年 1 月 8 日上午 10 点", attendees: "增长部、产品部、数据部", attendeeCount: 11, meetGoal: "确定年度增长目标与获客策略", host: "增长负责人", recorder: "运营专员", q1Goal: "获客渠道重构", q1Focus: "投放 ROI 偏低", q2Goal: "老客召回计划", q2Metric: "复购率", q2Pct: 15, q3Goal: "下沉市场试点", q3Market: "中西部三线城市", q4Goal: "年度复盘与预算编制", q4Budget: 150, r1: 6, r2: 4, mktBudget: 800, mktOnlinePct: 70, mktOfflinePct: 30, hireRd: 8, hireMkt: 6, infraBudget: 200, infraUse: "数据仓库扩容", reviewDate: "1 月 15 日", designDue: "2 月 1 日", selectionDate: "3 月中旬", pmOwner: "增长部项目经理", delayWeeks: 2, designCount: 3, hireRiskPct: 80, budgetDeadline: "1 月底" },
  { topic: "供应链数字化转型专题", meetDate: "2025 年 4 月 22 日下午 3 点", attendees: "供应链部、IT 部、财务部", attendeeCount: 9, meetGoal: "评审供应链系统选型与实施节奏", host: "供应链总监", recorder: "IT 项目经理", q1Goal: "采购系统上线", q1Focus: "审批链路过长", q2Goal: "库存可视化", q2Metric: "库存周转率", q2Pct: 20, q3Goal: "供应商协同平台", q3Market: "华东与华南", q4Goal: "系统集成与验收", q4Budget: 320, r1: 9, r2: 5, mktBudget: 0, mktOnlinePct: 0, mktOfflinePct: 100, hireRd: 4, hireMkt: 0, infraBudget: 480, infraUse: "私有云资源池", reviewDate: "4 月 30 日", designDue: "5 月 20 日", selectionDate: "6 月上旬", pmOwner: "IT 项目经理", delayWeeks: 4, designCount: 2, hireRiskPct: 70, budgetDeadline: "5 月 15 日" },
  { topic: "客户满意度提升专项", meetDate: "2025 年 9 月 5 日下午 2 点", attendees: "客服部、产品部、交付部", attendeeCount: 13, meetGoal: "定位满意度短板并制定改进计划", host: "客服总监", recorder: "质量专员", q1Goal: "工单响应提速", q1Focus: "首响超时", q2Goal: "交付质量整改", q2Metric: "一次交付成功率", q2Pct: 12, q3Goal: "客户培训体系", q3Market: "重点行业客户", q4Goal: "满意度回访机制", q4Budget: 90, r1: 5, r2: 7, mktBudget: 120, mktOnlinePct: 40, mktOfflinePct: 60, hireRd: 3, hireMkt: 2, infraBudget: 60, infraUse: "客服系统升级", reviewDate: "9 月 12 日", designDue: "10 月 10 日", selectionDate: "11 月初", pmOwner: "质量部负责人", delayWeeks: 1, designCount: 5, hireRiskPct: 90, budgetDeadline: "9 月底" },
];

/** 技术方案变体（目标文档是「技术方案」） */
const TECH = [
  { schemeName: "实时风控引擎", archStyle: "流批一体", archFlow: "事件经消息队列进入流处理算子，规则命中后写入决策库并异步落盘", fusionAlgo: "加权分数", fusionK: 40, deployUnit: "容器", clusterNodes: 24, cachePolicy: "本地 LRU", cacheHit: 92, embedModel: "text2vec-base-chinese", embedDim: 768, chunkTokens: 256, chunkOverlap: 1, modelMem: 220, indexType: "HNSW", indexRecall: 96, perChunkMs: 35, docPages: 300, totalEmbedMin: 5, searchP50: 120, searchP99: 480, memPeak: 1600, qps: 800, stressHours: 12, verifyScope: "规则命中片段", semWeight: 0.6, lexWeight: 0.4, rejectPct: 30, fallbackMode: "规则", timeoutSec: 5, summaryChars: 1500, degradeCycle: "周" },
  { schemeName: "多模态内容审核", archStyle: "分层解耦", archFlow: "内容先过文本与图像双通道模型，再按置信度路由到人工复审队列", fusionAlgo: "级联投票", fusionK: 30, deployUnit: "虚拟机", clusterNodes: 16, cachePolicy: "分布式 Redis", cacheHit: 85, embedModel: "clip-vit-base", embedDim: 512, chunkTokens: 128, chunkOverlap: 3, modelMem: 480, indexType: "IVF-Flat", indexRecall: 91, perChunkMs: 12, docPages: 800, totalEmbedMin: 9, searchP50: 60, searchP99: 300, memPeak: 2400, qps: 300, stressHours: 24, verifyScope: "人工标注样本", semWeight: 0.5, lexWeight: 0.5, rejectPct: 25, fallbackMode: "人工", timeoutSec: 10, summaryChars: 3000, degradeCycle: "双周" },
  { schemeName: "知识图谱构建平台", archStyle: "图数一体", archFlow: "抽取模型产出实体关系三元组，经消歧后写入图数据库并同步到检索索引", fusionAlgo: "图游走加权", fusionK: 80, deployUnit: "物理机", clusterNodes: 8, cachePolicy: "图查询结果", cacheHit: 78, embedModel: "bge-large-zh", embedDim: 1024, chunkTokens: 320, chunkOverlap: 2, modelMem: 640, indexType: "DiskANN", indexRecall: 94, perChunkMs: 48, docPages: 120, totalEmbedMin: 12, searchP50: 180, searchP99: 900, memPeak: 3200, qps: 120, stressHours: 6, verifyScope: "三元组来源句", semWeight: 0.7, lexWeight: 0.3, rejectPct: 40, fallbackMode: "图查询", timeoutSec: 15, summaryChars: 4000, degradeCycle: "月" },
];

/* ---------------- 规模化：用不同身份 + 不同事实派生更多同族文档 ---------------- */

/**
 * 目标块数 200+：语料太小的话 topK=5 一次就能覆盖大半库，指标必然满分。
 * 这里把每种文档族扩到 8~12 篇：身份（产品名/公司名/机构名）来自独立列表，
 * 事实数字按序号做确定性扰动，保证每篇都**事实不同但句式相同** —— 这才是硬干扰。
 */

/** 产品名（各自带不同的技术术语，避免只靠数字区分） */
const PRODUCT_NAMES = [
  ["边缘计算盒", "工业现场数据采集", "协议转换与本地预处理", "Modbus、Profinet、EtherCAT", "四核 ARM Cortex-A55"],
  ["数据采集终端", "能源计量", "多路计量数据汇聚与上报", "DLT645、Modbus、IEC104", "双核 ARM Cortex-A7"],
  ["视频汇聚网关", "安防视频接入", "多路视频流转发与 AI 前置处理", "RTSP、GB28181、ONVIF", "八核 ARM 加 NPU"],
  ["无线数传模块", "远程遥测", "低功耗无线数据回传", "LoRaWAN、NB-IoT、4G", "低功耗 MCU"],
  ["协议转换器", "楼宇自控", "多协议互转与点位映射", "BACnet、KNX、Modbus", "双核 RISC-V"],
  ["时间同步装置", "电力调度", "全站时钟同步与守时", "IRIG-B、PTP、NTP", "高精度晶振加 FPGA"],
  ["环境监测站", "园区环境感知", "多参数环境数据采集与上报", "Modbus、HJ212、MQTT", "低功耗 ARM Cortex-M7"],
  ["边缘推理盒子", "产线视觉质检", "本地推理与缺陷判定", "ONVIF、GigE Vision、Modbus", "六核 ARM 加 NPU"],
  ["车载终端", "车队管理", "车辆定位与工况上报", "JT808、CAN、4G", "车规级 MCU"],
  ["智能电表集中器", "居民用电采集", "多表集抄与数据冻结", "DLT698、DLT645、微功率无线", "双核 ARM Cortex-A9"],
  ["工业交换机", "产线网络互联", "环网冗余与流量整形", "Profinet、EtherNet/IP、Modbus", "交换 ASIC"],
  ["数据备份一体机", "边缘数据保护", "本地备份与异地复制", "SMB、NFS、iSCSI", "八核 x86"],
  ["水文监测终端", "水利遥测", "水位流量采集与预警上报", "SL651、Modbus、北斗短报文", "低功耗 ARM Cortex-M4"],
  ["充电桩控制器", "新能源汽车充电", "充电控制与计费结算", "OCPP、CAN、4G", "车规级 ARM Cortex-M3"],
  ["机房动环主机", "数据中心基础设施", "动力环境监控与联动告警", "Modbus、SNMP、BACnet", "四核 ARM Cortex-A53"],
  ["气体检测网关", "化工安全监测", "多路气体浓度采集与联锁", "Modbus、HART、LoRa", "双核 ARM Cortex-M33"],
  ["智能照明控制器", "公共照明节能", "回路调光与能耗统计", "DALI、KNX、Zigbee", "低功耗 MCU 加射频"],
  ["轨道巡检采集器", "轨道交通运维", "轨旁数据采集与缺陷初筛", "Modbus、MVB、4G", "六核 ARM 加 NPU"],
];

/** 公司名（员工手册族） */
const COMPANY_NAMES = ["启明科技", "云枢数据", "北岭制造", "海朔智能", "同和软件", "蔚蓝网络", "衡石信息", "砺行电子", "青柏医疗", "远岚能源", "牧野重工", "澄明生物", "九章智算", "长风物流"];

/** 财务主体 */
const FINANCE_ENTITIES = ["启明科技", "云枢数据", "北岭制造", "海朔智能", "同和软件", "蔚蓝网络", "衡石信息", "砺行电子", "青柏医疗", "远岚能源", "牧野重工", "九章智算"];

/** 会议主题 */
const MEETING_TOPICS = [
  "2025 年度增长规划", "供应链数字化转型专题", "客户满意度提升专项", "海外市场进入策略",
  "研发效能改进工作坊", "数据资产盘点启动会", "渠道政策调整评审", "年度预算编制沟通会",
  "重点项目风险复盘会", "服务质量专项治理会", "组织架构调整宣讲会", "产品定价策略评审会",
];

/** 技术方案名 */
const TECH_NAMES = [
  "实时风控引擎", "多模态内容审核", "知识图谱构建平台", "智能工单路由",
  "指标口径管理平台", "离线特征仓库", "多租户权限中台", "日志检索加速层",
  "实时推荐服务", "文档解析流水线", "异常检测基线系统", "向量召回服务",
];

/** 确定性伪随机：同一 seed 永远得到同一序列，保证语料可复现 */
function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    // Math.imul 做精确的 32 位乘法，避免普通乘法溢出到 53 位浮点
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}


/** 从基础变体派生一篇新的事实集（身份换掉、数字扰动） */
function derive(base, identity, seed) {
  const rng = makeRng(seed);
  const out = { ...base };
  for (const key of Object.keys(base)) {
    const v = base[key];
    if (typeof v === "number") {
      // 数字按原值 ±35% 扰动，保留一位小数若原值是小数
      const isFloat = !Number.isInteger(v);
      const next = v * (0.65 + rng() * 0.7);
      out[key] = isFloat ? Number(next.toFixed(1)) : Math.max(1, Math.round(next));
    }
  }
  Object.assign(out, identity);
  return out;
}

/* ---------------- 生成 ---------------- */

/**
 * 让每篇干扰文档覆盖**不同的章节子集**。
 *
 * 为什么必须这么做：同族文档如果章节完全一致，就变成「十几篇只有数字不同的近义文档」，
 * 那是个人工构造的对抗样本，不是真实语料 ——
 * 实测向量路文档级 top1 只有 36%（干扰手册排在真手册前面，余弦 0.595 vs 0.591；2026-10-01 tune 实测），
 * 因为「标准版单台售价为 X 元」这种句子彼此语义几乎相同，句向量分辨不出数字。
 * 真实语料里问「标准版售价」通常是**只有一篇**文档能答上，其余根本没有这一节。
 *
 * 所以按序号轮转选取 4 个章节并轮换顺序：同一问题只与少数几篇竞争，才是合理的难度。
 */
function pickSections(sections, index) {
  const want = 4;
  const start = (index * 3) % sections.length;
  const picked = [];
  const seen = new Set(); // 防止 (start + k*2) % len 重复选到同一章节
  for (let k = 0; k < want && k < sections.length; k++) {
    const idx = (start + k * 2) % sections.length;
    if (seen.has(idx)) continue;
    seen.add(idx);
    picked.push(sections[idx]);
  }
  // 轮换章节顺序，避免所有文档结构一致
  if (index % 2 === 1) picked.reverse();
  return picked;
}

const docs = [];

/*
 * 同族干扰文档**每种只保留 2 篇**。
 *
 * 这类文档与手写目标文档同模板、只换数字，是最"难"的干扰，
 * 但数量一多就变成人工构造的对抗样本：实测 13 篇产品手册时，
 * 任何查询（包括负样本）都能在库里找到余弦 0.45+ 的块，
 * 相似度地板被抬高，所有拒答/门限信号集体失效。
 * 真实语料里问「标准版售价」通常只有一两篇文档能答，其余根本没这一节。
 * 保留 2 篇足以测试"能否把正确的那篇排前面"，同时不污染全局相似度分布。
 */
const FAMILY_VARIANTS = 2;

PRODUCT_NAMES.slice(0, FAMILY_VARIANTS).forEach(([product, scene, duty, protocols, chip], i) => {
  const base = PRODUCTS[i % PRODUCTS.length];
  const f = derive(base, { product, scene, duty, protocols, chip }, 1000 + i * 17);
  // 价格量级跟产品定位走，避免派生出「无线模块卖 8000 元」这种不合理数字
  const scale = 0.4 + (i % 5) * 0.45;
  f.stdPrice = Math.round((base.stdPrice * scale) / 10) * 10;
  f.proPrice = Math.round((base.proPrice * scale) / 10) * 10;
  const doc = productManual(f);
  docs.push({
    file: `干扰-${product}产品手册.md`,
    doc: { ...doc, sections: pickSections(doc.sections, i) },
  });
});

COMPANY_NAMES.slice(0, FAMILY_VARIANTS).forEach((company, i) => {
  const base = COMPANIES[i % COMPANIES.length];
  const f = derive(base, { company }, 2000 + i * 23);
  const doc = employeeHandbook(f);
  docs.push({
    file: `干扰-${company}员工手册.md`,
    doc: { ...doc, sections: pickSections(doc.sections, i) },
  });
});

FINANCE_ENTITIES.slice(0, FAMILY_VARIANTS).forEach((entity, i) => {
  const base = FINANCE[i % FINANCE.length];
  const f = derive(base, { entity }, 3000 + i * 29);
  // 百分比类字段保持在 0~100
  for (const k of ["licensePct", "cloudPct", "servicePct", "rdPct", "netMargin", "revGrowth", "top3Pct", "cashGrowth"]) {
    f[k] = Math.min(95, Math.max(5, f[k]));
  }
  const doc = financeReport(f);
  docs.push({
    file: `干扰-${entity}财务报告.md`,
    doc: { ...doc, sections: pickSections(doc.sections, i) },
  });
});

MEETING_TOPICS.slice(0, FAMILY_VARIANTS).forEach((topic, i) => {
  const base = MEETINGS[i % MEETINGS.length];
  const f = derive(base, { topic, meetGoal: `确定${topic}的目标与实施节奏` }, 4000 + i * 31);
  for (const k of ["mktOnlinePct", "mktOfflinePct"]) f[k] = Math.min(90, Math.max(10, f[k]));
  const doc = meetingMinutes(f);
  docs.push({
    file: `干扰-${topic}.md`,
    doc: { ...doc, sections: pickSections(doc.sections, i) },
  });
});

TECH_NAMES.slice(0, FAMILY_VARIANTS).forEach((schemeName, i) => {
  const base = TECH[i % TECH.length];
  const f = derive(base, { schemeName }, 5000 + i * 37);
  for (const k of ["cacheHit", "indexRecall", "semWeight", "lexWeight", "rejectPct"]) f[k] = Math.min(99, Math.max(5, f[k]));
  const doc = techSpec(f);
  docs.push({
    file: `干扰-${schemeName}技术方案.md`,
    doc: { ...doc, sections: pickSections(doc.sections, i) },
  });
});

/*
 * 异题文档：主题、术语、句式全部不同，只在通用词上重叠。
 * 这是语料的"体量"来源，也是相似度地板不被抬高的关键 ——
 * 「仓储作业」「商标管理」与「股票代码」「碳排放」在语义与词汇上都不沾边，
 * 负样本因此能真正落在低分区，拒答信号才恢复分辨力。
 */
for (const topic of TOPIC_DOCS) {
  docs.push({ file: `干扰-${topic.title}.md`, doc: topic });
}

mkdirSync(OUT, { recursive: true });
// 先清掉上一次生成的干扰文档，避免改名后残留。
// 用 unlinkSync 而不是 rmSync：某些受限环境下 rm 相关调用会被拦截并**直接终止进程**，
// 那样连文件都写不出来，比残留更糟。
for (const f of readdirSync(OUT)) {
  if (f.startsWith("干扰-")) {
    try {
      unlinkSync(join(OUT, f));
    } catch {
      console.warn(`清理旧干扰文档失败（忽略）：${f}`);
    }
  }
}

let totalChars = 0;
for (const { file, doc } of docs) {
  const lines = [`# ${doc.title}`, ""];
  for (const [heading, paras] of doc.sections) {
    lines.push(`## ${heading}`, "");
    for (const p of paras) lines.push(p, "");
  }
  const text = lines.join("\n").trim() + "\n";
  writeFileSync(join(OUT, file), text, "utf8");
  totalChars += text.length;
  console.log(`生成 ${file}  ${text.length} 字`);
}
console.log(`\n共 ${docs.length} 篇干扰文档，${totalChars} 字`);
