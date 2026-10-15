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
    { id: "9ee583d9-6935-5ef0-b84d-095a260fea81", featured: true },
    { id: "0a37884b-3f5c-5740-9f4d-a4eae788e8bc", featured: true },
    { id: "2758c748-d121-541d-bbd7-b2b3908d6031", featured: true },
    { id: "4c2900f4-102c-5772-904c-b44aed20771f", featured: true },
    { id: "5195773b-f356-5ea2-9dd3-20dbf7bafac0" },
    { id: "e06510eb-793a-5f0b-b38f-9a49306ec852" },
    { id: "1a43530e-7874-5a1f-a83c-e0d94cf47d4c" },
    { id: "ba541c8f-373d-5a11-9286-bbfa59c5b032" },
    { id: "1fc6a374-339f-5755-90e8-653d87335ca2" },
    { id: "3f53ccd7-d85c-5c01-bb2e-b29705bf2451" },
    { id: "82f9aa7b-7839-529f-84e2-6ea916e50870", featured: true },
    { id: "2b73b8c2-fe22-52f6-a884-2c8b6f077f6f", featured: true }
  ],
  "dynamic": [
    { id: "bc904d9b-3a39-596f-9c95-4c0d7d0ee523", featured: true },
    { id: "42a094d9-9eac-59be-8f62-4ab86ce9e814", featured: true },
    { id: "06781124-993b-5fd3-b8e2-c7548e51e2d9", featured: true },
    { id: "0c5f25e6-a59a-5ed2-b7ce-7c603b44521f", featured: true },
    { id: "43643e49-62d0-575d-b22d-b2f814967dcc" },
    { id: "54aac642-634e-5169-9966-65e96185dc34" },
    { id: "075bf772-2a59-5f8b-874f-ffd5119cb78c", featured: true },
    { id: "119d45cf-4895-56fc-ab0f-5bd131fa32da" },
    { id: "adc38d9b-c044-5e06-9f2d-6d627e459ed5" },
    { id: "dfcfd1ca-ef24-5a46-8f6c-c839b5ecba26", featured: true },
    { id: "fd0d4cd3-d91b-5491-a33c-ba7795a98a50" },
    { id: "e467dd6e-0c20-512b-97a2-8051478ce7bb" }
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
