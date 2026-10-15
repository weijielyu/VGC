/* Comparison rows for the VGC project page.
 *
 * Each entry is one scene under static/videos/<id>/ holding two clips of the
 * same input: the source generator output and its VGC refinement.
 * `featured: true` rows are shown on index.html; more_results.html shows the
 * rest (`data-rest`), so no row appears on both pages.
 */

var VIDEO_BASE = "./static/videos";

var METHODS = [
  { file: "baseline.mp4",      label: "Generated Video" },
  { file: "refined_wan22.mp4", label: "VGC (Ours)", ours: true }
];

/* VG-Pairs rows (static/videos/vgpairs/<pair>/) show a different pair of clips. */
var GROUP_METHODS = {
  "vgpairs": [
    { file: "real.mp4",      label: "Real Video (Target)" },
    { file: "distorted.mp4", label: "Distorted Video (Input)" }
  ]
};

var RESULTS = {
  "vgpairs": [
    { id: "vgpairs/dl3dv_002400" },
    { id: "vgpairs/dl3dv_001650" },
    { id: "vgpairs/dl3dv_003153" }
  ],
  "static": [
    { id: "9ee583d9-6935-5ef0-b84d-095a260fea81", featured: true, caption: "VGC stabilizes the highway guardrail, which warps in the generated video." },
    { id: "0a37884b-3f5c-5740-9f4d-a4eae788e8bc", featured: true, caption: "VGC stabilizes the painting on the wall." },
    { id: "2758c748-d121-541d-bbd7-b2b3908d6031", featured: true, caption: "VGC stops the sofa from drifting during the camera turn." },
    { id: "4c2900f4-102c-5772-904c-b44aed20771f", featured: true, caption: "VGC keeps the mountain stationary while the camera dollies forward." },
    { id: "5195773b-f356-5ea2-9dd3-20dbf7bafac0", caption: "VGC keeps the mountain stationary as the camera moves along the slope." },
    { id: "e06510eb-793a-5f0b-b38f-9a49306ec852", caption: "VGC keeps the distant mountains static rather than drifting." },
    { id: "1a43530e-7874-5a1f-a83c-e0d94cf47d4c", caption: "VGC removes the drift of the building at the far end of the street." },
    { id: "ba541c8f-373d-5a11-9286-bbfa59c5b032", caption: "VGC preserves the rigid geometry of the elevated overpass." },
    { id: "1fc6a374-339f-5755-90e8-653d87335ca2", caption: "VGC fixes the distorted church pew and the picture sliding on the wall." },
    { id: "3f53ccd7-d85c-5c01-bb2e-b29705bf2451", caption: "VGC restores a coherent camera transition from the interior wall to the garden." },
    { id: "82f9aa7b-7839-529f-84e2-6ea916e50870", featured: true, caption: "VGC keeps the tree from vanishing mid-shot, while preserving a coherent camera trajectory." },
    { id: "2b73b8c2-fe22-52f6-a884-2c8b6f077f6f", featured: true, caption: "VGC removes the glass door that abruptly pops into view." }
  ],
  "dynamic": [
    { id: "bc904d9b-3a39-596f-9c95-4c0d7d0ee523", featured: true, caption: "VGC corrects the car's direction of travel." },
    { id: "42a094d9-9eac-59be-8f62-4ab86ce9e814", featured: true, caption: "VGC corrects the person's walking trajectory." },
    { id: "06781124-993b-5fd3-b8e2-c7548e51e2d9", featured: true, caption: "VGC restores correct car motion and a stable road sign." },
    { id: "0c5f25e6-a59a-5ed2-b7ce-7c603b44521f", featured: true, caption: "VGC distinguishes the parked silver car from the moving one and infers correct vehicle motion." },
    { id: "43643e49-62d0-575d-b22d-b2f814967dcc", caption: "VGC restores a coherent trajectory for the car." },
    { id: "54aac642-634e-5169-9966-65e96185dc34", caption: "VGC restores coherent motion across the busy traffic." },
    { id: "075bf772-2a59-5f8b-874f-ffd5119cb78c", featured: true, caption: "VGC removes artifacts on people moving through the market." },
    { id: "119d45cf-4895-56fc-ab0f-5bd131fa32da", caption: "VGC cleans up artifacts on the people and the background stall." },
    { id: "adc38d9b-c044-5e06-9f2d-6d627e459ed5", caption: "VGC restores correct motion of the person on the elevator." },
    { id: "dfcfd1ca-ef24-5a46-8f6c-c839b5ecba26", featured: true, caption: "VGC corrects the human motion seen through the window." },
    { id: "fd0d4cd3-d91b-5491-a33c-ba7795a98a50", caption: "VGC restores coherent motion across the human crowd." },
    { id: "e467dd6e-0c20-512b-97a2-8051478ce7bb", caption: "VGC removes the artifacts caused by the person's disappearance." }
  ]
};

function buildRow(item, methods) {
  var row = document.createElement("div");
  row.className = "result-group compare-row";

  if (item.caption) {
    var cap = document.createElement("div");
    cap.className = "has-text-centered result-label compare-caption";
    cap.textContent = item.caption;
    row.appendChild(cap);
  }

  var cols = document.createElement("div");
  cols.className = "compare-grid";
  methods.forEach(function (m) {
    var col = document.createElement("div");
    var label = document.createElement("div");
    label.className = "method-label" + (m.ours ? " is-ours" : "");
    label.textContent = m.label;

    var v = document.createElement("video");
    v.dataset.src = VIDEO_BASE + "/" + item.id + "/" + m.file;
    v.muted = true;
    v.playsInline = true;
    v.preload = "none";
    if (m.ours) v.dataset.ours = "1";

    col.appendChild(label);
    col.appendChild(v);
    cols.appendChild(col);
  });
  row.appendChild(cols);
  return row;
}

/* The two clips of a row have the same frame count but different frame
 * rates, so they are slowed to the duration of the longest one and restarted
 * together: every column then shows the same moment of the scene. */
function syncRow(row) {
  var videos = Array.prototype.slice.call(row.querySelectorAll("video"));
  var started = false;
  var visible = false;

  function restart() {
    videos.forEach(function (v) { v.currentTime = 0; });
    if (visible) videos.forEach(function (v) { v.play().catch(function () {}); });
  }

  function start() {
    var longest = Math.max.apply(null, videos.map(function (v) { return v.duration; }));
    var lead = videos[0];
    videos.forEach(function (v) {
      v.playbackRate = v.duration / longest;
      if (v.duration === longest) lead = v;
    });
    lead.addEventListener("ended", restart);
    started = true;
    restart();
  }

  function load() {
    var pending = videos.length;
    videos.forEach(function (v) {
      v.addEventListener("loadeddata", function () {
        if (--pending === 0) start();
      }, { once: true });
      v.preload = "auto";
      v.src = v.dataset.src;
    });
  }

  new IntersectionObserver(function (entries) {
    visible = entries[0].isIntersecting;
    if (visible && !videos[0].src) return load();
    if (!started) return;
    videos.forEach(function (v) {
      if (visible) v.play().catch(function () {}); else v.pause();
    });
  }, { rootMargin: "200px 0px" }).observe(row);
}

/* Fill every <div data-results="<group>"> with its rows. Add `data-featured`
 * to show only the rows marked `featured`, or `data-rest` for the others. */
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("[data-results]").forEach(function (host) {
    // data-results may list several groups ("static,dynamic"), shown in order.
    var items = [];
    host.dataset.results.split(",").forEach(function (name) {
      items = items.concat(RESULTS[name] || []);
    });
    if (host.hasAttribute("data-featured")) {
      items = items.filter(function (item) { return item.featured; });
    } else if (host.hasAttribute("data-rest")) {
      items = items.filter(function (item) { return !item.featured; });
    }
    items.forEach(function (item) {
      var row = buildRow(item, GROUP_METHODS[host.dataset.results] || METHODS);
      host.appendChild(row);
      syncRow(row);
    });
  });
});
