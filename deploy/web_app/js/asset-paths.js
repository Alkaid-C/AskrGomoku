/**
 * Resolve a deployable asset path.
 *
 * Development serves files directly from web_app/ and leaves the meta value
 * empty. The Cloudflare release builder fills it with an immutable,
 * content-addressed directory such as "assets/0123abcd.../".
 */
globalThis["GOMOKU_DEBUG"] = !document.querySelector(
    'meta[name="gomoku-asset-base"]')?.content;

function gomokuAssetUrl(relativePath) {
    const meta = document.querySelector('meta[name="gomoku-asset-base"]');
    const base = meta ? meta.content : '';
    return new URL(base + relativePath, document.baseURI).href;
}
