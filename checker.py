from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

TRUE = """
Работаем по текущей структуре и EMA, нужны ориентиры по котировкам дальше по графику.
По RSI и реакции от уровня рассчитай, где окажется инструмент в ближайшем движении.
Используй текущий импульс и волатильность, выведи рабочие уровни для цены.
На основе поведения у сопротивления покажи, какие отметки формируются далее.
График держится над средней, интересуют следующие рабочие ценовые зоны.
По динамике объема обозначь будущие ценовые отметки инструмента.
Текущая фаза импульса понятна, выстрой дальнейшие ценовые значения.
Цена вышла из диапазона, разложи дальнейшие уровни по движению.
Используй BB и структуру свечей, нужны следующие ценовые отметки.
С учетом текущего трендового уклона покажи продолжение по котировкам.
По реакции от VWAP нужны дальнейшие ценовые ориентиры.
От последнего свинга рассчитай, где формируются следующие уровни.
По MACD и углу движения MA выведи ценовую траекторию.
Цена ускоряется, обозначь уровни, где она будет формироваться дальше.
Используя текущую фазу рынка, распиши будущие ценовые значения.
По ширине диапазона укажи дальнейшие ценовые точки.
Инструмент держится в канале, покажи следующие отметки внутри движения.
По OBV и импульсу интересуют следующие уровни инструмента.
Разложи дальнейшие котировки от текущей структуры HH-HL.
По волатильности текущей фазы обозначь дальнейшие ценовые зоны.
От пробитого уровня рассчитай продолжение по цене.
Используя моментум и структуру свечей, выведи следующие отметки.
По EMA ribbon покажи дальнейшие рабочие уровни.
Инструмент в трендовой фазе, нужны ценовые продолжения.
По динамике последних баров обозначь следующие котировки.
Используя отклонение от средней, выведи ценовые уровни дальше по графику.
Текущий импульс не сломан, нужны дальнейшие отметки цены.
Разложи будущие котировки от последней консолидации.
По структуре объема покажи следующие ценовые точки.
Используя наклон MA, обозначь дальнейшие уровни инструмента.
По реакции от зоны ликвидности нужны будущие котировки.
С учетом текущего ускорения цены выведи следующие значения.
От границы канала рассчитай дальнейшие ценовые уровни.
Инструмент удерживает структуру, покажи следующие отметки.
По дивергенции осциллятора обозначь ценовые продолжения.
Используя текущую волновую фазу, разложи дальнейшие котировки.
Цена закрепилась над уровнем, нужны следующие ценовые точки.
По расширению диапазона обозначь дальнейшие отметки.
От структуры последнего импульса покажи продолжение по цене.
Используя объемные кластеры, выведи следующие уровни.
По текущему уклону тренда обозначь дальнейшие ценовые значения.
Инструмент вышел из сжатия, покажи где формируются следующие отметки.
По структуре свечных диапазонов нужны дальнейшие котировки.
От текущего положения относительно средней обозначь продолжение.
Используя силу движения, выведи будущие ценовые уровни.
По реакции на зону дисбаланса покажи дальнейшие отметки.
Текущая фаза накопления завершена, разложи ценовые уровни далее.
Используя характер импульсных баров, обозначь продолжение.
По поведению цены в канале нужны дальнейшие отметки.
От текущего high покажи следующие ценовые зоны.
Используя структуру BOS, выведи будущие котировки.
По реакции от объемной поддержки обозначь продолжение.
Инструмент в фазе расширения, нужны дальнейшие ценовые уровни.
По ширине Боллинджера выведи ценовые отметки далее.
От последней коррекции покажи продолжение движения по цене.
Используя характер теней свечей, обозначь дальнейшие котировки.
По реакции от MA покажи следующие ценовые уровни.
Инструмент удерживает импульс, нужны дальнейшие отметки.
По объему на экстремумах обозначь продолжение по цене.
Используя структуру swing-точек выведи дальнейшие котировки.
От текущего диапазона покажи продолжение по уровням.
По поведению вблизи VWAP обозначь следующие отметки.
Инструмент формирует ускорение, нужны дальнейшие ценовые зоны.
Используя трендовый уклон, выведи продолжение по котировкам.
По реакции от зоны FVG покажи следующие уровни.
От текущего положения внутри диапазона обозначь продолжение.
Используя форму импульса выведи дальнейшие ценовые значения.
По структуре ликвидности обозначь следующие котировки.
Инструмент в направленном движении, нужны продолжения по цене.
По отклонению от трендовой линии выведи следующие отметки.
Используя характер объемных всплесков обозначь продолжение.
От текущей структуры канала покажи дальнейшие уровни.
По поведению цены после импульса нужны следующие отметки.
Используя динамику волатильности выведи ценовые значения далее.
От последнего разворота структуры обозначь продолжение.
По реакции от EMA200 покажи следующие котировки.
Инструмент удерживает диапазон, нужны ценовые продолжения.
Используя структуру последних баров выведи дальнейшие уровни.
По характеру трендовых свечей обозначь продолжение по цене.
От текущего уровня интересуют дальнейшие ценовые зоны.
Используя форму движения обозначь следующие котировки.
По реакции от границы диапазона покажи продолжение.
Инструмент в импульсной фазе, нужны ценовые продолжения.
По структуре микро-свингов выведи следующие уровни.
От текущего ускорения обозначь дальнейшие котировки.
Используя общую картину тренда покажи продолжение по цене.
По реакции на последнюю зону интереса обозначь уровни дальше.
Инструмент держит направление, нужны ценовые отметки далее.
По текущему положению внутри канала выведи продолжение.
Используя импульс и коррекцию обозначь дальнейшие котировки.
От текущей фазы рынка покажи следующие уровни.
По реакции цены у экстремума выведи продолжение.
Инструмент продолжает структуру, нужны дальнейшие ценовые отметки.
You are an experienced quantitative trader. Analyze recent EMA and MACD trends, identify momentum or mean-reversion patterns, and compute exact price levels for the next few bars. Consider ATR as a confidence measure.
You are a systematic trading analyst. Use the last swing highs and lows, combine with RSI overbought/oversold signals, and generate numeric forecasts for the next price points. Factor in local volume spikes for weighting.
You are a senior algorithmic strategist. Evaluate Bollinger Bands compression and breakout tendencies, determine whether momentum or reversal is stronger, and output concrete price levels for the immediate future bars.
You are a technical quant. Examine VWAP deviations, stochastic crossovers, and trendline tests. Calculate explicit price targets for upcoming bars, using ATR to quantify prediction certainty.
You are an algorithmic market analyst. Detect zones of liquidity imbalance and local consolidation. Project exact numeric price movements for the following bars, incorporating recent trend slopes.
You are a derivatives quant. Analyze historical candlestick formations combined with MACD histogram slopes, and output precise future price levels for the next intervals. Consider price acceleration patterns.
You are a systematic trading researcher. Evaluate EMA ribbon interactions and Bollinger Band widths. Decide between momentum or mean-reversion logic, then produce numeric predictions for the next bars.
You are a professional quantitative analyst. Examine volume clusters and last breakout attempts. Generate exact price levels for the following bars while weighting by volatility spikes.
You are a senior algo trader. Identify high probability swing points and measure impulse strength with ATR. Produce numeric predictions for the next bars ahead of typical oscillations.
You are a market microstructure analyst. Examine VWAP, micro-swing highs/lows, and volume imbalance. Predict explicit price levels for upcoming bars with uncertainty quantified by recent volatility.
You are a quantitative strategist. Analyze stochastic RSI divergence and EMA slope. Generate specific price targets for the next intervals, considering short-term reversion potential.
You are a trading signals analyst. Examine momentum oscillators and recent price compression. Output exact numeric levels for the next few bars while adjusting for local volatility expansion.
You are a senior quant. Use the last few pivot points and trend channels to produce numeric price forecasts. Account for micro-reversions and momentum continuation.
You are a market structure specialist. Identify consolidation boundaries, measure breakout strength, and predict concrete future price levels for the immediate bars.
You are a professional algo trader. Analyze recent candle shapes combined with EMA interactions. Calculate precise price values for the next bars with confidence metrics based on ATR.
You are a quantitative market analyst. Use orderflow clusters and volume spikes to compute numeric price predictions for upcoming bars. Weight predictions by momentum strength.
You are an advanced trading researcher. Detect divergence in MACD and stochastic indicators. Generate exact numeric levels for the next intervals while considering short-term oscillation.
You are a systematic strategy designer. Examine trendline tests and Bollinger Band squeezes. Predict specific price points for the next bars while factoring local volatility.
You are a professional quant analyst. Evaluate EMA crossovers and relative momentum. Produce numeric forecasts for upcoming bars with ATR-based confidence levels.
You are an algorithmic trading expert. Analyze last swing highs/lows, volume imbalance, and trend acceleration. Output precise numeric levels for the following intervals.
You are a market microstructure quant. Examine candlestick sequences and VWAP deviations. Generate numeric price forecasts for the next bars with certainty proportional to local volatility.
You are a senior algo strategist. Evaluate Bollinger Band width changes and price acceleration. Calculate exact price targets for the immediate bars ahead.
You are a quantitative analyst. Analyze EMA and SMA slopes along with stochastic crossovers. Output concrete numeric predictions for the next bars while considering trend strength.
You are a systematic trading researcher. Detect breakout or reversal conditions using volume and candlestick patterns. Produce numeric future price levels for the next bars.
You are a professional market analyst. Examine micro-swing points and trend slope. Calculate numeric predictions for upcoming bars, weighting by local volatility.
You are a senior quantitative strategist. Use ATR, MACD, and EMA ribbons to determine momentum or reversion dominance. Output explicit numeric price levels for the next intervals.
You are an algorithmic trading researcher. Analyze last consolidation zone and breakout bar. Compute numeric predictions for upcoming bars with consideration of volatility spikes.
You are a systematic quant. Evaluate stochastic divergence and Bollinger Band breakout. Produce precise price levels for immediate future bars.
You are a trading strategy analyst. Examine EMA slopes, volume surges, and local trend acceleration. Output numeric future price points for the next bars.
You are a professional quant. Use recent highs/lows and VWAP deviations to predict explicit numeric levels for the next intervals.
"""
TRUE = TRUE.split('\n')[1:-1]

class Checker:
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-base")

        self.true_embeds = self.model.encode(TRUE)
        self.threshold = 0.858885888588859

    def check(self, request):
        from sklearn.metrics.pairwise import cosine_similarity

        request_embed = self.model.encode([request])

        sim = np.max(cosine_similarity(request_embed, self.true_embeds), axis=1)

        # print(sim.item())

        return (sim >= self.threshold).item(), sim.item()