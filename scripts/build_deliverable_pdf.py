"""Script para generar el documento PDF oficial de entrega:
TC5061_Experimentacion_MLFlow_BahamonRicardo.pdf

Cumple estrictamente con las directrices académicas y de formato:
- Tipografía: Times New Roman (Times-Roman), tamaño 11, interlineado 1.5 (leading=16.5).
- Extensión analítica: 2 cuartillas de texto y análisis riguroso (sin contar capturas ni anexos).
- Enlace al repositorio individual de GitHub.
- Comandos exactos de ejecución (por defecto y parametrizados).
- Tabla comparativa de las 6 corridas registradas en MLflow.
- Análisis profundo de resultados (mejor corrida, hiperparámetro influyente, limitaciones, reproducibilidad).
- Anexo de evidencias visuales con capturas en alta resolución de MLflow Tracking.
"""

import os
import shutil
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas de dos pasadas para calcular e imprimir dinámicamente 'Página X de Y'."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Encabezado superior
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(40, letter[1] - 32, letter[0] - 40, letter[1] - 32)
        self.drawString(
            40,
            letter[1] - 27,
            "TC5061 - MLOps | Semana 4: Experimentación Reutilizable con MLflow",
        )
        self.drawRightString(
            letter[0] - 40,
            letter[1] - 27,
            "Jesus Ricardo Bahamon Q. (A01840023)",
        )

        # Pie de página inferior
        self.line(40, 36, letter[0] - 40, 36)
        self.drawString(
            40,
            24,
            "Tecnológico de Monterrey | Maestría en Inteligencia Artificial Aplicada",
        )
        page_str = f"Página {self._pageNumber} de {total_pages}"
        self.drawRightString(letter[0] - 40, 24, page_str)
        self.restoreState()


def build_pdf(output_filename: str):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    # Estilos según especificación: Times New Roman, 11 pt, interlineado 1.5 (leading=16.5)
    body_style = ParagraphStyle(
        name="AcademicBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=16.5,
        textColor=colors.HexColor("#1A202C"),
        alignment=4,  # Justificado
        spaceAfter=5.5,
    )

    title_style = ParagraphStyle(
        name="AcademicTitle",
        fontName="Times-Bold",
        fontSize=14.5,
        leading=17.5,
        textColor=colors.HexColor("#0D3B66"),
        alignment=1,  # Centrado
        spaceAfter=2,
    )

    subtitle_style = ParagraphStyle(
        name="AcademicSubtitle",
        fontName="Times-Roman",
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        alignment=1,
        spaceAfter=7,
    )

    h1_style = ParagraphStyle(
        name="AcademicH1",
        fontName="Times-Bold",
        fontSize=11.5,
        leading=14.5,
        textColor=colors.HexColor("#0D3B66"),
        spaceBefore=6,
        spaceAfter=2.5,
        keepWithNext=True,
    )

    code_style = ParagraphStyle(
        name="AcademicCode",
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#F7FAFC"),
        borderPadding=3.5,
        spaceAfter=3.5,
    )

    meta_label_style = ParagraphStyle(
        name="MetaLabel",
        fontName="Times-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1A365D"),
    )

    meta_value_style = ParagraphStyle(
        name="MetaValue",
        fontName="Times-Roman",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
    )

    table_header_style = ParagraphStyle(
        name="TableHeader",
        fontName="Times-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        name="TableCell",
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
        alignment=1,
    )

    table_cell_left = ParagraphStyle(
        name="TableCellLeft",
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
        alignment=0,
    )

    caption_style = ParagraphStyle(
        name="FigureCaption",
        fontName="Times-Italic",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceBefore=2.5,
        spaceAfter=7,
    )

    story = []

    # =========================================================================
    # PÁGINA 1: CUARTILLA 1 - ARQUITECTURA, PARAMETRIZACIÓN E INSTRUMENTACIÓN
    # =========================================================================

    story.append(
        Paragraph("TC5061 - Operaciones de Aprendizaje Automático (MLOps)", title_style)
    )
    story.append(
        Paragraph(
            "Actividad Semana 4: Código de Experimentación Reutilizable Instrumentado con MLFlow",
            subtitle_style,
        )
    )

    # Tabla de metadatos académicos y enlace a GitHub
    meta_data = [
        [
            Paragraph("<b>Estudiante:</b>", meta_label_style),
            Paragraph("Jesus Ricardo Bahamon Quimbaya (Matrícula: A01840023)", meta_value_style),
            Paragraph("<b>Profesor:</b>", meta_label_style),
            Paragraph("Dr. Gerardo Reyes", meta_value_style),
        ],
        [
            Paragraph("<b>Institución:</b>", meta_label_style),
            Paragraph("Tecnológico de Monterrey - MNA", meta_value_style),
            Paragraph("<b>Fecha:</b>", meta_label_style),
            Paragraph("Octubre 2026", meta_value_style),
        ],
        [
            Paragraph("<b>Repositorio GitHub:</b>", meta_label_style),
            Paragraph(
                "<font color='#0D3B66'><u>https://github.com/ricardo-bahamon/TC5061-MLFlow-Experimentation</u></font>",
                meta_value_style,
            ),
            Paragraph("<b>Estructura:</b>", meta_label_style),
            Paragraph("Cookiecutter Data Science", meta_value_style),
        ],
    ]
    t_meta = Table(meta_data, colWidths=[90, 222, 58, 162])
    t_meta.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(t_meta)
    story.append(Spacer(1, 4))

    # 1. Migración y Organización del Código
    story.append(Paragraph("1. Migración Modular y Estructura Cookiecutter Data Science", h1_style))
    p1_text = (
        "En esta actividad individual se transformó el flujo experimental y de entrenamiento del modelo "
        "base sobre el conjunto de datos <i>Wine Quality (Red Wine)</i> en un pipeline de producción "
        "modular, reutilizable y parametrizado (<code>train.py</code>). El cuaderno Jupyter original "
        "(<code>notebooks/Wine_EDA.ipynb</code>) se reservó exclusivamente para análisis exploratorio "
        "y visualización preliminar. Siguiendo rigurosamente la plantilla <b>Cookiecutter Data Science</b>, "
        "el repositorio organiza el código en módulos con responsabilidades únicas y desacopladas: "
        "<code>src/data/make_dataset.py</code> implementa la detección automática de separadores (coma o punto y coma), "
        "limpieza y eliminación de duplicados; <code>src/features/build_features.py</code> aísla las covariables "
        "y realiza la partición estratificada de entrenamiento y prueba (80/20); y "
        "<code>src/models/train_model.py</code> construye un <code>Pipeline</code> de scikit-learn compuesto por "
        "<code>StandardScaler</code> y <code>LogisticRegression</code>, evitando fugas de información (<i>data leakage</i>)."
    )
    story.append(Paragraph(p1_text, body_style))

    # 2. Parametrización CLI y Comandos de Ejecución
    story.append(Paragraph("2. Parametrización de Ejecución con Argumentos de Línea de Comandos", h1_style))
    p2_text = (
        "El script <code>train.py</code> implementa una interfaz CLI mediante <code>argparse</code> que expone cuatro "
        "hiperparámetros de modelado y la semilla determinista: la fuerza de regularización "
        "(<code>--C</code>, tipo float), el algoritmo de optimización (<code>--solver</code>: lbfgs, saga, liblinear), "
        "la norma de penalización (<code>--penalty</code>: l1, l2), el número máximo de iteraciones (<code>--max-iter</code>) "
        "y la semilla pseudoaleatoria (<code>--seed</code>). Además, permite configurar la ruta de datos, la variable "
        "objetivo, el tamaño de prueba y los pliegues de validación cruzada. Cualquier colaborador o evaluador puede reproducir "
        "el flujo completo con valores predeterminados sin modificar el código fuente."
    )
    story.append(Paragraph(p2_text, body_style))

    # Comandos exactos
    story.append(Paragraph("<b>Comando exacto de ejecución con valores por defecto:</b>", body_style))
    story.append(Paragraph("python train.py", code_style))

    story.append(Paragraph("<b>Comando exacto para la ejecución con hiperparámetros personalizados (Mejor Corrida):</b>", body_style))
    cmd_custom = (
        "python train.py --experiment-name \"TC5061_Wine_Quality_Experimentation\" "
        "--run-name \"run_5_lasso_l1_saga\" --C 1.0 --solver saga --penalty l1 --max-iter 1500 --seed 42"
    )
    story.append(Paragraph(cmd_custom, code_style))

    # 3. Instrumentación con MLflow Tracking
    story.append(Paragraph("3. Instrumentación con MLflow Tracking", h1_style))
    p3_text = (
        "Cada ejecución del script se gestiona de forma atómica dentro de un contexto <code>mlflow.start_run()</code>, "
        "agrupada bajo el experimento común <code>TC5061_Wine_Quality_Experimentation</code>. La instrumentación registra: "
        "(a) <b>Hiperparámetros y metadatos</b> con <code>mlflow.log_params</code> (C, solver, penalty, max_iter, seed, scaler, dimensiones); "
        "(b) <b>Métricas de evaluación</b> con <code>mlflow.log_metrics</code> (Accuracy, macro/weighted F1-Score, macro Precision, macro Recall, "
        "junto con el promedio y dispersión de exactitud en 5 pliegues de validación cruzada); "
        "(c) <b>Artefactos de diagnóstico</b> con <code>mlflow.log_artifact</code> (matriz de confusión gráfica en PNG a 300 DPI, "
        "reporte textual por clase <code>classification_report.txt</code> y resumen en JSON <code>metrics_summary.json</code>); "
        "y (d) <b>Modelo serializado</b> con <code>mlflow.sklearn.log_model</code>, guardando el pipeline completo con firma de "
        "inferencia, ejemplos de entrada y dependencias para reproducibilidad y despliegue."
    )
    story.append(Paragraph(p3_text, body_style))

    # =========================================================================
    # PÁGINA 2: CUARTILLA 2 - ANÁLISIS DE RESULTADOS, HIPERPARÁMETROS Y LIMITACIONES
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("4. Comparación Experimental y Análisis de Resultados", h1_style))
    p4_intro = (
        "Se ejecutó una serie sistemática de seis corridas experimentales en MLflow variando la intensidad de regularización "
        "(C ∈ {0.01, 0.1, 1.0, 10.0}), la norma de penalización (L1 vs L2) y el optimizador. A continuación se presentan "
        "los resultados consolidados registrados y comparados en la interfaz de MLflow Tracking:"
    )
    story.append(Paragraph(p4_intro, body_style))

    # Tabla comparativa de corridas
    table_data = [
        [
            Paragraph("Corrida / Nombre", table_header_style),
            Paragraph("C", table_header_style),
            Paragraph("Solver", table_header_style),
            Paragraph("Norma", table_header_style),
            Paragraph("Seed", table_header_style),
            Paragraph("Exactitud (Test)", table_header_style),
            Paragraph("F1 Weighted", table_header_style),
            Paragraph("CV Exactitud (5-Folds)", table_header_style),
        ],
        [
            Paragraph("Run 1: Baseline L2", table_cell_left),
            Paragraph("1.0", table_cell_style),
            Paragraph("lbfgs", table_cell_style),
            Paragraph("L2", table_cell_style),
            Paragraph("42", table_cell_style),
            Paragraph("0.5809", table_cell_style),
            Paragraph("0.5541", table_cell_style),
            Paragraph("0.5943 ± 0.023", table_cell_style),
        ],
        [
            Paragraph("Run 2: Alta Reg. (C=0.01)", table_cell_left),
            Paragraph("0.01", table_cell_style),
            Paragraph("lbfgs", table_cell_style),
            Paragraph("L2", table_cell_style),
            Paragraph("42", table_cell_style),
            Paragraph("0.5662", table_cell_style),
            Paragraph("0.5113", table_cell_style),
            Paragraph("0.5658 ± 0.024", table_cell_style),
        ],
        [
            Paragraph("Run 3: Reg. Moderada (C=0.1)", table_cell_left),
            Paragraph("0.1", table_cell_style),
            Paragraph("lbfgs", table_cell_style),
            Paragraph("L2", table_cell_style),
            Paragraph("42", table_cell_style),
            Paragraph("0.5846", table_cell_style),
            Paragraph("0.5549", table_cell_style),
            Paragraph("0.5842 ± 0.021", table_cell_style),
        ],
        [
            Paragraph("Run 4: Baja Reg. (C=10.0)", table_cell_left),
            Paragraph("10.0", table_cell_style),
            Paragraph("lbfgs", table_cell_style),
            Paragraph("L2", table_cell_style),
            Paragraph("42", table_cell_style),
            Paragraph("0.5809", table_cell_style),
            Paragraph("0.5571", table_cell_style),
            Paragraph("0.5897 ± 0.027", table_cell_style),
        ],
        [
            Paragraph("<b>Run 5: Lasso L1 SAGA (Mejor)</b>", table_cell_left),
            Paragraph("1.0", table_cell_style),
            Paragraph("saga", table_cell_style),
            Paragraph("<b>L1</b>", table_cell_style),
            Paragraph("42", table_cell_style),
            Paragraph("<b>0.5919</b>", table_cell_style),
            Paragraph("<b>0.5655</b>", table_cell_style),
            Paragraph("0.5861 ± 0.024", table_cell_style),
        ],
        [
            Paragraph("Run 6: Réplica Determinista", table_cell_left),
            Paragraph("1.0", table_cell_style),
            Paragraph("lbfgs", table_cell_style),
            Paragraph("L2", table_cell_style),
            Paragraph("42", table_cell_style),
            Paragraph("0.5809", table_cell_style),
            Paragraph("0.5541", table_cell_style),
            Paragraph("0.5943 ± 0.023", table_cell_style),
        ],
    ]
    t_runs = Table(table_data, colWidths=[130, 30, 42, 38, 30, 80, 75, 107])
    t_runs.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D3B66")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#EBF8FF")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_runs)
    story.append(Spacer(1, 5))

    # Análisis del mejor modelo
    story.append(Paragraph("<b>Identificación de la Mejor Configuración:</b>", h1_style))
    p_best = (
        "La configuración con el desempeño superior en el conjunto de prueba independiente fue la <b>Corrida 5 "
        "(run_5_lasso_l1_saga)</b>, alcanzando una <b>Exactitud de 0.5919 (59.19%)</b> y un <b>F1-Score ponderado "
        "de 0.5655</b>, superando al modelo base por 1.10 puntos porcentuales en exactitud y 1.14 puntos en F1 ponderado. "
        "Este modelo utiliza regularización Lasso (penalización L1 con C=1.0) resuelto mediante el algoritmo de descenso "
        "de gradiente estocástico promedio adaptado (<code>saga</code>), permitiendo convergencia estricta en 1500 iteraciones."
    )
    story.append(Paragraph(p_best, body_style))

    # Hiperparámetro más influyente
    story.append(Paragraph("<b>Hiperparámetro Más Influyente y Sensibilidad:</b>", h1_style))
    p_param = (
        "El hiperparámetro con mayor impacto estructural fue el <b>tipo de penalización (penalty: L1 vs L2)</b>, seguido "
        "estrechamente por el inverso de regularización (<code>C</code>). La penalización L1 induce dispersión (<i>sparsity</i>) "
        "anulando los coeficientes de covariables colineales o con bajo aporte explicativo (como densidad o acidez fija), "
        "lo cual atenúa el ruido en las decisiones multiclase entre calidades contiguas (5, 6 y 7). "
        "Por otro lado, una regularización desmedida (<b>C=0.01</b> en Run 2) provocó un <b>subajuste severo (underfitting)</b>, "
        "desplomando la exactitud a 0.5662 y el F1 ponderado a 0.5113 debido a que contrajo excesivamente los pesos de "
        "variables fisicoquímicas críticas como el alcohol y los sulfatos."
    )
    story.append(Paragraph(p_param, body_style))

    # Reproducibilidad y fijación de semilla
    story.append(Paragraph("<b>Documentación de Reproducibilidad y Dependencias:</b>", h1_style))
    p_repro = (
        "La reproducibilidad científica se verificó empíricamente mediante la <b>Corrida 6</b>, la cual reprodujo de manera "
        "idéntica los argumentos de la Corrida 1 (C=1.0, lbfgs, L2, seed=42). Como se observa en la tabla, ambas corridas "
        "arrojaron métricas exactamente idénticas hasta el último decimal (Exactitud = 0.580882, CV = 0.594326), "
        "validando que la fijación de la semilla en el particionado, el escalado y el optimizador garantiza determinismo absoluto. "
        "El entorno computacional se encuentra completamente fijado en <code>requirements.txt</code> con versiones exactas "
        "(<code>mlflow==3.16.1</code>, <code>scikit-learn==1.7.2</code>, <code>pandas==2.2.3</code>, <code>numpy==2.2.6</code>)."
    )
    story.append(Paragraph(p_repro, body_style))

    # Limitaciones observadas
    story.append(Paragraph("<b>Limitaciones del Modelo:</b>", h1_style))
    p_limits = (
        "Se identificaron dos limitaciones centrales: (1) <b>Fuerte desbalance de clases en la calidad del vino</b>: más del "
        "82% de las muestras corresponden a las clases 5 y 6, mientras que las calificaciones extremas (3 y 8) cuentan con "
        "menos de 15 ejemplos en total. Esto provoca que el modelo lineal ignore las clases minoritarias (precisión y recall "
        "cercanos a 0 en clases 3 y 8), reflejándose en una brecha notable entre el F1 Macro (0.276) y el F1 Weighted (0.565). "
        "(2) <b>Suposición de linealidad</b>: Las interacciones organolépticas entre pH, acidez volátil y alcohol presentan "
        "fronteras no lineales que una regresión logística no puede capturar sin expansiones polinomiales o árboles de decisión."
    )
    story.append(Paragraph(p_limits, body_style))

    # =========================================================================
    # PÁGINA 3: ANEXO DE EVIDENCIAS VISUALES - TABLA COMPARATIVA Y PARALLEL COORD
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("Anexo A: Evidencias de Experimentación en MLflow Tracking", title_style))
    story.append(
        Paragraph(
            "Capturas oficiales de la interfaz web de MLflow Tracking demostrando el registro de corridas y comparación visual.",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 4))

    # Evidencia 1: Tabla comparativa de corridas
    fig1_path = Path("reports/figures/runs_table_direct.png")
    if fig1_path.exists():
        story.append(Paragraph("<b>Figura 1: Tabla General de Corridas Registradas en el Experimento MLFlow</b>", h1_style))
        story.append(
            Paragraph(
                "Vista de la pestaña <i>Model training</i> en MLflow (puerto 5050), exhibiendo las seis corridas completadas con sus "
                "respectivos identificadores, estado, hiperparámetros (C, solver, penalty, seed) y métricas de evaluación.",
                body_style,
            )
        )
        img1 = Image(str(fig1_path), width=515, height=205)
        story.append(img1)
        story.append(
            Paragraph(
                "<i>Captura 1: Vista general de las seis corridas en el experimento 'TC5061_Wine_Quality_Experimentation'.</i>",
                caption_style,
            )
        )

    story.append(Spacer(1, 6))

    # Evidencia 2: Vista de comparación paralela de métricas vs parámetros
    fig2_path = Path("reports/figures/mlflow_compare_view.png")
    if fig2_path.exists():
        story.append(Paragraph("<b>Figura 2: Vista Comparativa de Corridas y Gráficos de Coordenadas Paralelas</b>", h1_style))
        story.append(
            Paragraph(
                "Herramienta interactiva de comparación de MLflow contrastando las configuraciones evaluadas. El gráfico de coordenadas "
                "paralelas ilustra el impacto directo del parámetro C y la penalización en el balance entre exactitud y F1-score.",
                body_style,
            )
        )
        img2 = Image(str(fig2_path), width=515, height=210)
        story.append(img2)
        story.append(
            Paragraph(
                "<i>Captura 2: Vista comparativa avanzada en MLflow con gráfico de coordenadas paralelas y contraste de hiperparámetros.</i>",
                caption_style,
            )
        )

    # =========================================================================
    # PÁGINA 4: ANEXO DE EVIDENCIAS VISUALES - DETALLE DE CORRIDA Y ARTEFACTOS
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("Anexo B: Detalle de Corrida Óptima y Artefactos Registrados", title_style))
    story.append(
        Paragraph(
            "Inspección detallada de la mejor corrida ('run_5_lasso_l1_saga') con sus métricas y artefactos serializados.",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 4))

    # Evidencia 3: Detalle de la corrida óptima
    fig3_path = Path("reports/figures/mlflow_run_detail.png")
    if fig3_path.exists():
        story.append(Paragraph("<b>Figura 3: Detalle de la Corrida Óptima (run_5_lasso_l1_saga)</b>", h1_style))
        story.append(
            Paragraph(
                "Ficha técnica de la corrida en MLflow mostrando el Run ID (<code>2d4513f7bca241faaaa622a8959775dd</code>), "
                "la totalidad de hiperparámetros registrados con <code>log_param</code> y las métricas computadas con <code>log_metric</code>.",
                body_style,
            )
        )
        img3 = Image(str(fig3_path), width=515, height=205)
        story.append(img3)
        story.append(
            Paragraph(
                "<i>Captura 3: Inspección de metadata, hiperparámetros y métricas de la corrida óptima en MLflow.</i>",
                caption_style,
            )
        )

    story.append(Spacer(1, 6))

    # Evidencia 4: Explorador de artefactos con la matriz de confusión previsualizada
    fig4_clean = Path("reports/figures/mlflow_artifacts_view_clean.png")
    fig4_fallback = Path("reports/figures/mlflow_artifact_cm_expanded.png")
    fig4_path = fig4_clean if fig4_clean.exists() else fig4_fallback
    if fig4_path.exists():
        story.append(Paragraph("<b>Figura 4: Explorador de Artefactos con Previsualización de Matriz de Confusión</b>", h1_style))
        story.append(
            Paragraph(
                "Pestaña de artefactos en MLflow mostrando el directorio <code>evaluation_artifacts/</code> desplegado con la "
                "matriz de confusión (<code>confusion_matrix.png</code>) renderizada en el visor interactivo, el reporte "
                "textual y el resumen de métricas.",
                body_style,
            )
        )
        img4 = Image(str(fig4_path), width=515, height=210)
        story.append(img4)
        story.append(
            Paragraph(
                "<i>Captura 4: Explorador de artefactos en MLflow exhibiendo la matriz de confusión y artefactos de evaluación.</i>",
                caption_style,
            )
        )

    # Compilar documento con NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f" PDF generado exitosamente en: {output_filename}")


if __name__ == "__main__":
    out_dir = Path("/Users/ricardobahamon/Documents/Maestria/Trimestre_4/Operaciones_AI/Semana_4")
    out_file = out_dir / "TC5061_Experimentacion_MLFlow_BahamonRicardo.pdf"
    build_pdf(str(out_file))

    # También guardar una copia en el directorio de reportes del proyecto
    project_copy = Path("reports") / "TC5061_Experimentacion_MLFlow_BahamonRicardo.pdf"
    shutil.copy2(out_file, project_copy)
    print(f" Copia guardada en: {project_copy}")
