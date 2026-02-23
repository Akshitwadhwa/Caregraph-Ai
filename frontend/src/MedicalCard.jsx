import React from 'react';
import './MedicalCard.css';
import { AlertTriangle, Info, CheckCircle, BrainCircuit } from 'lucide-react';

export default function MedicalCard({ data }) {
    if (!data) return null;

    return (
        <div className="medical-card">
            {/* Disclaimer Section */}
            {data.disclaimer && (
                <div className="mc-section mc-disclaimer">
                    <AlertTriangle className="mc-icon text-warning" size={20} />
                    <div>
                        <strong>Disclaimer:</strong> {data.disclaimer}
                    </div>
                </div>
            )}

            {/* Rationale Section */}
            {data.rationale && (
                <div className="mc-section mc-rationale">
                    <BrainCircuit className="mc-icon text-primary" size={20} />
                    <div className="mc-content">
                        <h4 className="mc-title">Clinical Rationale</h4>
                        <p>{data.rationale}</p>
                    </div>
                </div>
            )}

            {/* OK Report / Final Output */}
            {data.ok_report && (
                <div className="mc-section mc-report">
                    <CheckCircle className="mc-icon text-success" size={20} />
                    <div className="mc-content">
                        <h4 className="mc-title">Summary & Recommendation</h4>
                        <p>{data.ok_report}</p>
                    </div>
                </div>
            )}
        </div>
    );
}
