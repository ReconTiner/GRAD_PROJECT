require([
    "esri/config",
    "esri/Map",
    "esri/layers/CSVLayer",
    "esri/views/MapView",
    "esri/widgets/Legend",
    "esri/widgets/ScaleBar",
    "esri/widgets/Sketch",
    "esri/Graphic",
    "esri/layers/GraphicsLayer",
    "esri/geometry/geometryEngine",],
    function (
        esriConfig,
        Map,
        CSVLayer,
        MapView,
        Legend,
        ScaleBar,
        Sketch,
        Graphic,
        GraphicsLayer,
        geometryEngine) {

        esriConfig.apiKey = "AAPKf467f63bec014c8fa266ae623ee9f475ecFFIrG6H2k1U_gS-9Eygn9BggsmVW7pPxUaYfiUVSmMcTtFGjEvIqa59lIsHACP";

        const template = {
            title: "{place}",
            content: "Magnitude {mag} {type} hit {place} on {time}."
        };

        // The heatmap renderer assigns each pixel in the view with
        // an intensity value. The ratio of that intensity value
        // to the maxPixel intensity is used to assign a color
        // from the continuous color ramp in the colorStops property
        const renderer = {
            type: "heatmap",
            field: "Badly",
            colorStops: [
                { color: "rgba(63, 40, 102, 0)", ratio: 0 },
                { color: "#472b77", ratio: 0.083 },
                { color: "#4e2d87", ratio: 0.166 },
                { color: "#563098", ratio: 0.249 },
                { color: "#5d32a8", ratio: 0.332 },
                { color: "#6735be", ratio: 0.415 },
                { color: "#7139d4", ratio: 0.498 },
                { color: "#7b3ce9", ratio: 0.581 },
                { color: "#853fff", ratio: 0.664 },
                { color: "#a46fbf", ratio: 0.747 },
                { color: "#c29f80", ratio: 0.83 },
                { color: "#e0cf40", ratio: 0.913 },
                { color: "#ffff00", ratio: 1 }
            ],
            maxPixelIntensity: 300,
            minPixelIntensity: 0
        };

        const layer = new CSVLayer({
            url: "../sample.csv",
            title: "建议施药区域",
            copyright: "Recon Tiner",
            popupTemplate: template,
            renderer: renderer
        });

        const map = new Map({
            basemap: "arcgis-imagery",
            layers: [layer]
        });

        const view = new MapView({
            container: "main-box",
            center: [117.5526377222, 34.3100582222],
            zoom: 18,
            map: map
        });

        view.ui.add(
            new Legend({
                view: view
            }),
            "bottom-left"
        );

        // 添加比例尺工具
        const scalebar = new ScaleBar({
            view: view,
            unit: "metric"
        });
        view.ui.add(scalebar, "bottom-right");

        // 添加工具图层
        const graphicsLayer = new GraphicsLayer();
        map.add(graphicsLayer);

        const sketch = new Sketch({
            layer: graphicsLayer,
            view: view,
            availableCreateTools: ["polyline", "polygon"],
            creationMode: "update",
            updateOnGraphicClick: true,
            visibleElements: {
                createTools: {
                    point: false,
                    circle: false
                },
                selectionTools: {
                    "lasso-selection": false,
                    "rectangle-selection": false,
                },
                settingsMenu: false,
                undoRedoMenu: false
            }
        });
        view.ui.add(sketch, "top-right");

        const temp = document.getElementById("temp")
        view.ui.add(temp, "manual");

        sketch.on("update", (e) => {
            const geometry = e.graphics[0].geometry;

            if (e.state === "start") {
                switchType(geometry);
            }

            if (e.state === "complete") {
                // graphicsLayer.remove(graphicsLayer.graphics.getItemAt(0));
                // measurements.innerHTML = null;
            }

            if (
                e.toolEventInfo &&
                (e.toolEventInfo.type === "scale-stop" ||
                    e.toolEventInfo.type === "reshape-stop" ||
                    e.toolEventInfo.type === "move-stop")
            ) {
                switchType(geometry);
            }

        });

        // 计算面积
        var geodesicArea;
        var planarArea;
        function getArea(polygon) {
            geodesicArea = geometryEngine.geodesicArea(polygon, "square-meters");
            planarArea = geometryEngine.planarArea(polygon, "square-meters");
            temp.innerHTML =
                "<b>Geodesic area</b>:  " + geodesicArea.toFixed(2) + " m\xB2" + " |   <b>Planar area</b>: " + planarArea.toFixed(2) + "  m\xB2<br>";
        }

        function getLength(line) {
            const geodesicLength = geometryEngine.geodesicLength(line, "meters");
            const planarLength = geometryEngine.planarLength(line, "meters");
            measurements.innerHTML =
                "<b>Geodesic length</b>:  " + geodesicLength.toFixed(2) + " m" + " |   <b>Planar length</b>: " + planarLength.toFixed(2) + "  m";
        }

        function switchType(geom) {
            switch (geom.type) {
                case "polygon":
                    getArea(geom);
                    break;
                case "polyline":
                    getLength(geom);
                    break;
                default:
                    console.log("No value found");
            }
        }

        // 记录测量信息
        count = 0;
        var totalGeodesicArea = 0;
        var totalPlanarArea = 0;
        const recordButton = document.getElementById("record");
        recordButton.onclick = function () {
            totalGeodesicArea += parseFloat(geodesicArea);
            totalPlanarArea += parseFloat(planarArea);
            // 记录合计结果
            let GeoArea = totalGeodesicArea.toFixed(2);
            let PlaArea = totalPlanarArea.toFixed(2);
            document.getElementById("total").innerHTML =
                "<b>Total Geodesic area</b>:  " + GeoArea + " m\xB2" + "<b><br>Total Planar area</b>: " + PlaArea + "  m\xB2";
            // 在表格中插入每一次记录的结果
            count += 1;
            var tab = document.getElementById("record-result-tab");
            var td1 = document.createElement("td");
            var td2 = document.createElement("td");
            var td3 = document.createElement("td");
            var td4 = document.createElement("td");
            // 赋值
            td1.innerHTML = count;
            td2.innerHTML = geodesicArea.toFixed(2) + " m\xB2";
            td3.innerHTML = planarArea.toFixed(2) + "  m\xB2";
            var a = document.createElement("a");
            a.innerHTML = "删除";
            a.href = "#";
            a.style.color = "rgb(90, 125, 176)";
            // 删除事件
            a.onclick = function () {
                let tr = this.parentNode.parentNode;
                totalGeodesicArea -= parseFloat(tr.cells[1].innerHTML);
                totalPlanarArea -= parseFloat(tr.cells[2].innerHTML);
                let GeoArea = totalGeodesicArea.toFixed(2);
                let PlaArea = totalPlanarArea.toFixed(2);
                document.getElementById("total").innerHTML =
                    "<b>Total Geodesic area</b>:  " + GeoArea + " m\xB2" + "<b><br>Total Planar area</b>: " + PlaArea + "  m\xB2";
                graphicsLayer.remove(graphicsLayer.graphics.getItemAt(parseInt(tr.cells[0].innerHTML) - 1));
                this.parentNode.parentNode.remove();
            }
            td4.appendChild(a);
            var tr = document.createElement("tr");
            tr.appendChild(td1);
            tr.appendChild(td2);
            tr.appendChild(td3);
            tr.appendChild(td4);
            tab.appendChild(tr);
        }

        // 删除上一条信息
        // const deleteButton = document.getElementById("a");
        // deleteButton.onclick = function (this) {
        //     // 在地图上删除矢量要素
        //     // graphicsLayer.remove(graphicsLayer.graphics.getItemAt(count - 1));            
        //     // gg= Table2.rows[aa].cells[cel].childNodes[0].value; 得到 aa 行  cel 列   0为 第一个控件的值。
        //     this.parentNode.parentNode.remove();
        //     // totalGeodesicArea -= parseFloat(geodesicArea);
        //     // totalPlanarArea -= parseFloat(planarArea);
        //     // let GeoArea = totalGeodesicArea.toFixed(2);
        //     // let PlaArea = totalPlanarArea.toFixed(2);
        //     // document.getElementById("total").innerHTML =
        //     //     "<b>Total Geodesic area</b>:  " + GeoArea + " m\xB2" + "<b><br>Total Planar area</b>: " + PlaArea + "  m\xB2";
        //     count -= 1;
        // }
    });