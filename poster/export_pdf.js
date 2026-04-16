#!/usr/bin/env node
// Exports poster/index.html as an A0 landscape PDF.
// Usage: node poster/export_pdf.js

const { chromium } = require('playwright');
const path = require('path');
const fs   = require('fs');

// A0 landscape at 96 dpi: 1189mm × 841mm → ~4493 × 3178 px
const A0_W_MM  = 1189;
const A0_H_MM  = 841;
const PX_PER_MM = 96 / 25.4;           // ≈ 3.78
const A0_W_PX  = Math.round(A0_W_MM * PX_PER_MM); // 4493
const A0_H_PX  = Math.round(A0_H_MM * PX_PER_MM); // 3178
const POSTER_W = 1189;                  // native poster width in px
const ZOOM     = A0_W_PX / POSTER_W;   // ≈ 3.779

const OUT = path.resolve(__dirname, '..', 'poster_A0.pdf');

const INJECT = `
<style id="a0-override">
@page { size: ${A0_W_MM}mm ${A0_H_MM}mm; margin: 0; }
@media print {
  html, body {
    width:  ${A0_W_PX}px !important;
    height: ${A0_H_PX}px !important;
    overflow: hidden !important;
  }
  .poster {
    zoom: ${ZOOM};
    box-shadow: none !important;
    margin: 0 !important;
  }
}
</style>
`;

(async () => {
  // Write a temporary HTML file so local images resolve correctly
  let html = fs.readFileSync(path.resolve(__dirname, 'index.html'), 'utf8');
  html = html.replace('</head>', INJECT + '</head>');
  const tmpPath = path.resolve(__dirname, '_export_tmp.html');
  fs.writeFileSync(tmpPath, html);

  const browser = await chromium.launch();
  const page    = await browser.newPage();

  await page.setViewportSize({ width: A0_W_PX, height: A0_H_PX });
  await page.goto('file://' + tmpPath, { waitUntil: 'networkidle' });

  await page.pdf({
    path:            OUT,
    width:           `${A0_W_MM}mm`,
    height:          `${A0_H_MM}mm`,
    printBackground: true,
    preferCSSPageSize: true,
  });

  await browser.close();
  fs.unlinkSync(tmpPath);
  console.log('PDF written to:', OUT);
})();
