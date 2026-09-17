#include "AntialiasResizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

	// Accumulatore di riga della passata verticale: outW*3 interi a 32 bit.
	// thread_local perche' Resize e' const e una sola istanza serve tutti i
	// thread di preprocessing. Allocato una volta per thread, mai nel percorso
	// caldo. L'alternativa sarebbe passarlo come parametro accanto a scratch,
	// al prezzo di cambiare la firma di Resize.
	int* VerticalAccumulator(std::size_t elems)
	{
		thread_local std::vector<int> acc;
		if (acc.size() < elems) acc.resize(elems);
		return acc.data();
	}

} // namespace

AntialiasResizer::AntialiasResizer(int inW, int inH, int outW, int outH) :
	inW_(inW),
	inH_(inH),
	outW_(outW),
	outH_(outH),
	hK_(0),
	vK_(0),
	identity_(inW == outW && inH == outH)
{
	PrecomputeCoeffs(inW, outW, hBounds, hWeights, hK_);
	PrecomputeCoeffs(inH, outH, vBounds, vWeights, vK_);
}

void AntialiasResizer::Resize(const cv::Mat& src, cv::Mat& dst, cv::Mat& scratch) const
{
	CV_Assert(src.type() == CV_8UC3);
	CV_Assert(Matches(src.cols, src.rows));

	const int inH = inH_, inW = inW_;
	const int outW = outW_, outH = outH_;

	// Identity fast path. With inSize == outSize the triangle filter collapses
	// to weights [1, 0] anchored on bounds[o] == o for every output pixel, so
	// the two passes below reduce to a pixel-for-pixel copy. Aliasing instead of
	// copying keeps this at zero cost; dst is read-only for the caller.
	if (identity_) {
		dst = src;
		return;
	}

	// Passata orizzontale: [inH x inW] -> [inH x outW]
	// L'ordine resta o esterno / t interno: la sorgente e' a passo variabile in
	// o (bounds[o] avanza di ~scale), quindi invertire produrrebbe una gather
	// che non si vettorizza. Qui il guadagno viene dall'aritmetica intera e
	// dall'avanzamento incrementale di px.
	CV_Assert(scratch.rows == inH_ && scratch.cols == outW_ && scratch.type() == CV_8UC3);
	cv::Mat& hpass = scratch;
	for (int y = 0; y < inH; ++y) {
		const uint8_t* srow = src.ptr<uint8_t>(y);
		uint8_t* hrow = hpass.ptr<uint8_t>(y);
		for (int o = 0; o < outW; ++o) {
			const int s = hBounds[o];
			const int avail = (std::min)(hK_, inW - s);
			const std::int16_t* w = &hWeights[static_cast<size_t>(o) * hK_];
			int a0 = 0, a1 = 0, a2 = 0;
			const uint8_t* px = srow + static_cast<size_t>(s) * 3;
			for (int t = 0; t < avail; ++t, px += 3) {
				const int wt = w[t];
				a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
			}
			uint8_t* op = hrow + static_cast<size_t>(o) * 3;
			op[0] = RoundShiftClipU8(a0);
			op[1] = RoundShiftClipU8(a1);
			op[2] = RoundShiftClipU8(a2);
		}
	}

	// Passata verticale: [inH x outW] -> [outH x outW]
	// Qui i loop SONO invertiti. Nella forma precedente (x esterno, t interno)
	// ogni pixel di uscita saltava fra vK_ righe distanti fra loro: accesso a
	// passo lungo, nessun prefetch, nessuna vettorizzazione, e ptr<uint8_t>()
	// ricalcolato dentro il loop su x. Accumulando una riga per volta, il loop
	// interno diventa una MAC su interi con entrambi gli operandi sequenziali,
	// che MSVC auto-vettorizza senza intrinseci.
	dst.create(outH, outW, CV_8UC3);
	const int n3 = outW * 3;
	int* acc = VerticalAccumulator(static_cast<size_t>(n3));

	for (int o = 0; o < outH; ++o) {
		const int s = vBounds[o];
		const int avail = (std::min)(vK_, inH - s);
		const std::int16_t* w = &vWeights[static_cast<size_t>(o) * vK_];

		if (avail <= 0) {                 // non dovrebbe accadere
			std::memset(acc, 0, sizeof(int) * static_cast<size_t>(n3));
		}
		else {
			// La prima riga assegna invece di accumulare: evita il memset.
			const uint8_t* srow = hpass.ptr<uint8_t>(s);
			const int w0 = w[0];
			for (int i = 0; i < n3; ++i) acc[i] = w0 * srow[i];

			for (int t = 1; t < avail; ++t) {
				const int wt = w[t];
				if (wt == 0) continue;    // la coda dei pesi e' spesso nulla
				const uint8_t* row = hpass.ptr<uint8_t>(s + t);
				for (int i = 0; i < n3; ++i) acc[i] += wt * row[i];
			}
		}

		uint8_t* drow = dst.ptr<uint8_t>(o);
		for (int i = 0; i < n3; ++i) drow[i] = RoundShiftClipU8(acc[i]);
	}
}

uint8_t AntialiasResizer::RoundShiftClipU8(int acc)
{
	// Arrotondamento al piu' vicino senza floating point: aggiunge mezzo LSB
	// prima dello shift. Sostituisce std::floor(v + 0.5), che veniva chiamata
	// tre volte per pixel di uscita.
	int r = (acc + (1 << (kWeightShift - 1))) >> kWeightShift;
	if (r < 0) r = 0;
	if (r > 255) r = 255;
	return static_cast<uint8_t>(r);
}

bool AntialiasResizer::Matches(int srcW, int srcH) const
{
	return srcW == inW_ && srcH == inH_;
}

int AntialiasResizer::OutW() const
{
	return outW_;
}

int AntialiasResizer::OutH() const
{
	return outH_;
}

void AntialiasResizer::PrecomputeCoeffs(int inSize, int outSize, std::vector<int>& bounds, std::vector<std::int16_t>& weights, int& ksize)
{
	const double scale = static_cast<double>(inSize) / static_cast<double>(outSize);
	const double filterscale = scale >= 1.0 ? scale : 1.0;
	const double support = 1.0 * filterscale;
	const double ss = 1.0 / filterscale;

	ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
	bounds.assign(outSize, 0);
	weights.assign(static_cast<size_t>(outSize) * ksize, 0);

	// I pesi si calcolano ancora in doppia precisione: questo codice gira una
	// volta all'avvio, e partire dai valori esatti riduce l'errore di
	// quantizzazione. Solo il risultato finale diventa intero.
	std::vector<double> wd(static_cast<size_t>(ksize), 0.0);

	for (int o = 0; o < outSize; ++o)
	{
		const double center = (o + 0.5) * scale;
		int xmin = static_cast<int>(center - support + 0.5);
		if (xmin < 0) xmin = 0;
		int xmax = static_cast<int>(center + support + 0.5);
		if (xmax > inSize) xmax = inSize;

		int n = xmax - xmin;
		if (n > ksize) n = ksize;          // la geometria garantisce n <= ksize

		double total = 0.0;
		for (int t = 0; t < n; ++t)
		{
			double w = 1.0 - std::abs((xmin + t - center + 0.5) * ss);
			if (w < 0.0) w = 0.0;
			wd[t] = w;
			total += w;
		}

		if (total > 0.0)
			for (int t = 0; t < n; ++t) wd[t] /= total;

		// Quantizzazione a Q14 con correzione della somma. Senza la correzione
		// la somma dei pesi interi finisce a 16383 o 16385 invece di 16384, e
		// ogni pixel esce scalato di un fattore costante: l'immagine si
		// schiarisce o si scurisce in modo uniforme di qualche LSB. Su una
		// anomaly map diventa uno spostamento sistematico degli score,
		// difficile da attribuire a posteriori. L'errore si scarica sul peso
		// dominante, dove pesa di meno in termini relativi.
		std::int16_t* row = &weights[static_cast<size_t>(o) * ksize];
		int isum = 0;
		int imax = 0;
		for (int t = 0; t < n; ++t)
		{
			const int wi = static_cast<int>(std::lround(wd[t] * kWeightOne));
			row[t] = static_cast<std::int16_t>(wi);
			isum += wi;
			if (wd[t] > wd[imax]) imax = t;
		}
		if (n > 0)
			row[imax] = static_cast<std::int16_t>(row[imax] + (kWeightOne - isum));

		bounds[o] = xmin;
	}
}

cv::Mat AntialiasResizer::MakeScratch() const
{
	cv::Mat m(inH_, outW_, CV_8UC3);
	std::memset(m.data, 0, m.total() * m.elemSize());
	return m;
}

cv::Mat AntialiasResizer::MakeDestination() const
{
	if (identity_) return cv::Mat();   // il fast path aliasa src, il buffer sarebbe sprecato
	cv::Mat m(outH_, outW_, CV_8UC3);
	std::memset(m.data, 0, m.total() * m.elemSize());   // commit delle pagine
	return m;
}